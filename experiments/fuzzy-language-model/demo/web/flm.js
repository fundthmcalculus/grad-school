/* In-browser inference for the fuzzy language models (FuzzyDelta mixer + first-order TSK FFN, tied embedding).
 *
 * Works as a plain <script> (exposes window.FLM) and under Node (module.exports) for the parity test.
 * Nothing here talks to a server: weights come from ./models/<id>.{json,bin}, the byte-level BPE from ./tokenizer.json.
 *
 *   const m = new FLM.Model(manifest, arrayBuffer);   m.reset();   const logits = m.step(tokenId, {explain: true});
 *
 * The recurrence is the model's own step form (flm/models.py `_DeltaMixer.step_forward`); the rule explanation is the
 * exact direct-path decomposition of flm/rule_dist.py: every FFN rule adds gate * s * pi[rule] to the logits, where
 * s = 1/rms(final state) and pi[rule] = (conseq[rule] * norm_weight) . E^T is a fixed vocabulary profile.
 */
(function (root) {
  'use strict';

  // ------------------------------------------------------------------ tokenizer (HF byte-level BPE)
  function byteToUnicode() {
    const bs = [];
    for (let i = 33; i <= 126; i++) bs.push(i);
    for (let i = 161; i <= 172; i++) bs.push(i);
    for (let i = 174; i <= 255; i++) bs.push(i);
    const cs = bs.slice();
    let n = 0;
    for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
    const enc = new Array(256), dec = new Map();
    bs.forEach((b, i) => { enc[b] = String.fromCodePoint(cs[i]); dec.set(String.fromCodePoint(cs[i]), b); });
    return { enc, dec };
  }
  const B2U = byteToUnicode();
  const PRE = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

  class Tokenizer {
    constructor(json) {
      this.vocab = json.model.vocab;                       // token string -> id
      this.idToTok = [];
      for (const [t, i] of Object.entries(this.vocab)) this.idToTok[i] = t;
      this.rank = new Map();
      json.model.merges.forEach((m, i) => this.rank.set(Array.isArray(m) ? m.join(' ') : m, i));
      const sp = (json.added_tokens || []).find((a) => a.special);
      this.eot = sp ? sp.id : -1;
      this.eotChar = sp ? sp.content : null;
      this.cache = new Map();
    }
    _bpe(word) {                                           // word: string of mapped chars
      if (this.cache.has(word)) return this.cache.get(word);
      let parts = Array.from(word);
      while (parts.length > 1) {
        let best = -1, bi = -1;
        for (let i = 0; i < parts.length - 1; i++) {
          const r = this.rank.get(parts[i] + ' ' + parts[i + 1]);
          if (r !== undefined && (best < 0 || r < best)) { best = r; bi = i; }
        }
        if (bi < 0) break;
        parts.splice(bi, 2, parts[bi] + parts[bi + 1]);
      }
      const ids = parts.map((p) => this.vocab[p]);
      this.cache.set(word, ids);
      return ids;
    }
    encode(text) {
      const out = [], utf8 = new TextEncoder();
      const segs = this.eotChar ? text.split(this.eotChar) : [text];   // the special token is atomic
      segs.forEach((seg, k) => {
        if (k > 0) out.push(this.eot);
        for (const m of seg.matchAll(PRE)) {
          const mapped = Array.from(utf8.encode(m[0])).map((b) => B2U.enc[b]).join('');
          for (const id of this._bpe(mapped)) out.push(id);
        }
      });
      return out;
    }
    // text of one token as the Python demo shows it
    piece(id) {
      if (id === this.eot) return '\n<|eot|>\n';
      const bytes = Array.from(this.idToTok[id]).map((c) => B2U.dec.get(c));
      return new TextDecoder('utf-8', { fatal: false }).decode(Uint8Array.from(bytes));
    }
  }

  // ------------------------------------------------------------------ weights
  function half(h) {                                       // IEEE half -> float
    const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
    if (e === 31) return f ? NaN : s * Infinity;
    return s * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  function readTensors(manifest, buf) {
    const T = {}, fp16 = manifest.dtype === 'float16', sz = fp16 ? 2 : 4;
    const dv = new DataView(buf);
    for (const t of manifest.tensors) {
      const n = t.shape.reduce((a, b) => a * b, 1), a = new Float32Array(n);
      if (t.q) {                                          // int8: q x one float32 scale per row of a matrix (one per tensor for a vector); see demo/quantize_model.py
        const rows = t.shape.length >= 2 ? t.shape[0] : 1, per = n / rows, q = new Int8Array(buf, t.offset, n);
        const scale = new Float32Array(rows);
        for (let r = 0; r < rows; r++) { const sc = (scale[r] = dv.getFloat32(t.scale_offset + r * 4, true)); for (let k = r * per, e = k + per; k < e; k++) a[k] = q[k] * sc; }
        T[t.name] = { data: a, shape: t.shape, q, scale };
        continue;
      } else for (let i = 0; i < n; i++) a[i] = fp16 ? half(dv.getUint16(t.offset + i * 2, true)) : dv.getFloat32(t.offset + i * 4, true);
      T[t.name] = { data: a, shape: t.shape };
    }
    return T;
  }
  const EPS = 1.1920929e-7;                                // torch.finfo(float32).eps, nn.RMSNorm(eps=None)
  const sigmoid = (z) => 1 / (1 + Math.exp(-z));
  function expScale(kind, dim) { return kind === 'sum' ? 1 : kind === 'mean' ? 1 / dim : Math.pow(dim, -0.5); }

  // ------------------------------------------------------------------ model
  class Model {
    constructor(manifest, buf) {
      const c = manifest.config;
      if (c.mixer !== 'fuzzydelta' || c.ffn !== 'tsk' || c.decay !== 'data' || !(c.shortconv > 1) || c.emb_dim)
        throw new Error('this engine implements the fuzzydelta + TSK-FFN + data-decay + shortconv models only');
      this.manifest = manifest; this.cfg = c;
      const W = (this.W = readTensors(manifest, buf));
      this.V = c.vocab_size; this.d = c.d_model; this.H = c.n_heads; this.dh = this.d / this.H; this.R = c.n_rules;
      this.L = c.n_layers; this.width = c.shortconv; this.maxLen = c.max_len;
      this.E = W.emb.data; this.nw = W.norm.data;
      this.layers = [];
      for (let l = 0; l < this.L; l++) {
        const g = (k) => W['l' + l + '.' + k].data;
        const fr = W['l' + l + '.f.centers'].shape[0], da = W['l' + l + '.f.centers'].shape[1];
        const mA = new Float32Array(g('m.log_width').length), fA = new Float32Array(g('f.log_width').length);
        g('m.log_width').forEach((v, i) => { mA[i] = Math.exp(-2 * v); });
        g('f.log_width').forEach((v, i) => { fA[i] = Math.exp(-2 * v); });
        this.layers.push({
          n1: g('n1'), conv: g('conv'), mc: g('m.centers'), ma: mA, qkv: g('m.qkv'), out: g('m.out'),
          gw: g('m.gate_w'), gb: g('m.gate_b'), bw: g('m.beta_w'), bb: g('m.beta_b'),
          n2: g('n2'), fc: g('f.centers'), fa: fA, conseq: g('f.conseq'), ante: g('f.ante'),
          sw: g('f.slope_w'), sb: g('f.slope_b'), FR: fr, da,
          mScale: expScale(c.exp_norm, this.dh), fScale: expScale(c.ffn_exp_norm || c.exp_norm, da),
          order: c.ffn_order, normNone: c.ffn_norm === 'none',
        });
      }
      this.pi = null;
      this.reset();
    }
    reset() {
      this.pos = 0;
      this.state = this.layers.map(() => ({
        S: new Float32Array(this.H * this.dh * this.R),
        ring: new Float32Array(this.width * this.d),
      }));
    }
    _rms(x, w, out) {
      let ss = 0; for (let i = 0; i < x.length; i++) ss += x[i] * x[i];
      const r = 1 / Math.sqrt(ss / x.length + EPS);
      for (let i = 0; i < x.length; i++) out[i] = x[i] * r * w[i];
      return r;
    }
    // one token in, the next-token logits out. opts.explain: also return the FFN gates (per layer) and the final scale s
    step(tok, opts) {
      const { d, H, dh, R, width } = this, t = this.pos;
      const x = Float32Array.from(this.E.subarray(tok * d, tok * d + d));
      const h = new Float32Array(d), hc = new Float32Array(d), qkv = new Float32Array(3 * d), y = new Float32Array(d);
      const gates = [];
      for (let l = 0; l < this.L; l++) {
        const P = this.layers[l], st = this.state[l];
        this._rms(x, P.n1, h);
        const slot = ((t % width) + width) % width;
        st.ring.set(h, slot * d);
        for (let c = 0; c < d; c++) {                      // causal depthwise conv; weight[width-1] multiplies the current h
          let a = 0;
          for (let k = 0; k < width; k++) {
            const tt = t - (width - 1) + k;
            if (tt >= 0) a += P.conv[c * width + k] * st.ring[((tt % width) + width) % width * d + c];
          }
          hc[c] = a;
        }
        for (let o = 0; o < 3 * d; o++) { let a = 0; const r = o * d; for (let i = 0; i < d; i++) a += P.qkv[r + i] * hc[i]; qkv[o] = a; }
        const fq = new Float32Array(R), fk = new Float32Array(R);
        for (let hd = 0; hd < H; hd++) {
          this._phi(P, hd, qkv, hd * dh, fq);              // q: component 0
          this._phi(P, hd, qkv, d + hd * dh, fk);          // k: component 1
          let za = P.gb[hd], zb = P.bb[hd];
          for (let i = 0; i < d; i++) { za += P.gw[hd * d + i] * hc[i]; zb += P.bw[hd * d + i] * hc[i]; }
          const alpha = sigmoid(za), beta = sigmoid(zb);
          const S = st.S, base = hd * dh * R;
          for (let j = 0; j < dh; j++) {
            const row = base + j * R, vj = qkv[2 * d + hd * dh + j];
            let pred = 0;
            for (let r = 0; r < R; r++) { S[row + r] *= alpha; pred += S[row + r] * fk[r]; }
            const e = beta * (vj - pred);
            let yo = 0;
            for (let r = 0; r < R; r++) { S[row + r] += e * fk[r]; yo += S[row + r] * fq[r]; }
            y[hd * dh + j] = yo;
          }
        }
        for (let o = 0; o < d; o++) { let a = 0; const r = o * d; for (let i = 0; i < d; i++) a += P.out[r + i] * y[i]; x[o] += a; }
        // TSK feed-forward
        const n = new Float32Array(d); this._rms(x, P.n2, n);
        const u = new Float32Array(P.da);
        for (let j = 0; j < P.da; j++) { let a = 0; for (let i = 0; i < d; i++) a += P.ante[j * d + i] * n[i]; u[j] = a; }
        const g = new Float32Array(P.FR);
        let mx = -Infinity;
        for (let r = 0; r < P.FR; r++) {
          let q = 0; for (let j = 0; j < P.da; j++) { const df = u[j] - P.fc[r * P.da + j]; q += P.fa[r * P.da + j] * df * df; }
          g[r] = -0.5 * P.fScale * q; if (g[r] > mx) mx = g[r];
        }
        if (P.normNone) for (let r = 0; r < P.FR; r++) g[r] = Math.exp(g[r]);
        else { let s = 0; for (let r = 0; r < P.FR; r++) { g[r] = Math.exp(g[r] - mx); s += g[r]; } for (let r = 0; r < P.FR; r++) g[r] /= s; }
        if (P.order === 1) for (let r = 0; r < P.FR; r++) { let a = P.sb[r]; for (let i = 0; i < d; i++) a += P.sw[r * d + i] * n[i]; g[r] *= a; }
        for (let r = 0; r < P.FR; r++) { const gr = g[r]; if (gr !== 0) for (let i = 0; i < d; i++) x[i] += gr * P.conseq[r * d + i]; }
        if (opts && opts.explain) gates.push(g);
      }
      const hn = new Float32Array(d);
      const s = this._rms(x, this.nw, hn);
      const logits = new Float32Array(this.V);
      for (let v = 0; v < this.V; v++) { let a = 0; const r = v * d; for (let i = 0; i < d; i++) a += this.E[r + i] * hn[i]; logits[v] = a; }
      this.pos++;
      return opts && opts.explain ? { logits, gates, s } : { logits };
    }
    _phi(P, hd, qkv, off, out) {                            // fuzzy partition: softmax_r(-0.5 * scale * sum_j a (u - c)^2)
      const { dh, R } = this; let mx = -Infinity;
      for (let r = 0; r < R; r++) {
        let q = 0; const b = (hd * R + r) * dh;
        for (let j = 0; j < dh; j++) { const df = qkv[off + j] - P.mc[b + j]; q += P.ma[b + j] * df * df; }
        out[r] = -0.5 * P.mScale * q; if (out[r] > mx) mx = out[r];
      }
      let s = 0; for (let r = 0; r < R; r++) { out[r] = Math.exp(out[r] - mx); s += out[r]; }
      for (let r = 0; r < R; r++) out[r] /= s;
    }
    // vocabulary profiles of every FFN rule (computed on first use)
    profiles() {
      if (this.pi) return this.pi;
      const { d, V } = this;
      this.pi = this.layers.map((P) => {
        const pi = new Float32Array(P.FR * V), cw = new Float32Array(d);
        for (let r = 0; r < P.FR; r++) {
          for (let i = 0; i < d; i++) cw[i] = P.conseq[r * d + i] * this.nw[i];
          for (let v = 0; v < V; v++) { let a = 0; for (let i = 0; i < d; i++) a += cw[i] * this.E[v * d + i]; pi[r * V + v] = a; }
        }
        return pi;
      });
      return this.pi;
    }
    // membership functions of one FFN rule read directly from the weights (no data statistics needed)
    ruleShape(layer, rule) {
      const P = this.layers[layer], dims = [];
      for (let j = 0; j < P.da; j++) {
        const w = 1 / Math.sqrt(P.fa[rule * P.da + j]);      // log_width -> width
        dims.push({ dim: j, center: P.fc[rule * P.da + j], sigma: w / Math.sqrt(P.fScale) });
      }
      return dims;
    }
    mixerRule(layer, head, rule) {
      const P = this.layers[layer], b = (head * this.R + rule) * this.dh, c = [], w = [];
      for (let j = 0; j < this.dh; j++) { c.push(P.mc[b + j]); w.push(1 / Math.sqrt(P.ma[b + j])); }
      return { layer, head, rule, center: c, width: w };
    }
  }


  // ------------------------------------------------------------------ the all-fuzzy model (every block component a fuzzy inference system)
  // A line-for-line port of allfuzzy/allfuzzy_lm.py: gain controllers instead of RMSNorm, rule-gated low-rank maps instead of q/k/v/out projections,
  // six-rule TSK gates, a fuzzy associative memory, a feed-forward bank of Gaussian rules on the raw input (layer 0 reads two tokens), and a token
  // table whose tail pieces are fuzzy sets over 12 prototypes.  Same interface as Model (reset, step, profiles, ruleShape, mixerRule).
  function gaussLogits(u, c, a, R, D, scale) {          // FUZZIFY + FIRE: log of each rule's product-t-norm firing strength
    const out = new Float64Array(R);
    const D4 = D - (D % 4);
    for (let r = 0; r < R; r++) {
      let q0 = 0, q1 = 0, q2 = 0, q3 = 0, j = 0; const b = r * D;
      for (; j < D4; j += 4) {                         // four independent sums: the engine cannot vectorise, but it can overlap them
        const d0 = u[j] - c[b + j], d1 = u[j + 1] - c[b + j + 1], d2 = u[j + 2] - c[b + j + 2], d3 = u[j + 3] - c[b + j + 3];
        q0 += a[b + j] * d0 * d0; q1 += a[b + j + 1] * d1 * d1; q2 += a[b + j + 2] * d2 * d2; q3 += a[b + j + 3] * d3 * d3;
      }
      for (; j < D; j++) { const df = u[j] - c[b + j]; q0 += a[b + j] * df * df; }
      out[r] = -0.5 * scale * (q0 + q1 + q2 + q3);
    }
    return out;
  }
  function softmaxF(v) { let mx = -Infinity; for (const x of v) if (x > mx) mx = x; let s = 0; const o = new Float64Array(v.length); for (let i = 0; i < v.length; i++) { o[i] = Math.exp(v[i] - mx); s += o[i]; } for (let i = 0; i < o.length; i++) o[i] /= s; return o; }
  // ------------------------------------------------------------------ SIMD kernels (WebAssembly, hand-assembled: no toolchain, 3 functions)
  // gauss(u, c, a, R, D, out):  out[r] = sum_j a[r,j] (u[j] - c[r,j])^2           (the rules' squared distances; D a multiple of 4)
  // matvec(M, v, R, D, out):    out[r] = sum_j M[r,j] v[j]                           (slopes of the rules, the output scores)
  // wsum(M, s, R, D, out):      out[j] = sum_r s[r] M[r,j]                           (the consequents, weighted by the rule strengths)
  // All pointers are byte offsets into one shared memory, all numbers float32 (four lanes at a time).  Without SIMD support the model uses its plain loops.
  const Wasm = (() => {
    const leb = (n) => { const o = []; do { let b = n & 0x7f; n >>>= 7; if (n) b |= 0x80; o.push(b); } while (n); return o; };
    const sleb = (n) => { const o = []; for (;;) { const b = n & 0x7f; n >>= 7; if ((n === 0 && !(b & 0x40)) || (n === -1 && (b & 0x40))) { o.push(b); return o; } o.push(b | 0x80); } };
    const str = (s) => [...leb(s.length), ...[...s].map((c) => c.charCodeAt(0))];
    const vec = (items) => [...leb(items.length), ...items.flat()];
    const section = (id, body) => [id, ...leb(body.length), ...body];
    const GET = (i) => [0x20, ...leb(i)], SET = (i) => [0x21, ...leb(i)], I32C = (n) => [0x41, ...sleb(n)];
    const ADD = [0x6a], MUL = [0x6c], SUB = [0x6b], GEU = [0x4f], LTU = [0x49];
    const VLOAD = [0xfd, 0x00, 0x00, 0x00], VSTORE = [0xfd, 0x0b, 0x00, 0x00], VZERO = [0xfd, 0x0c, ...new Array(16).fill(0)];
    const F32LOAD = [0x2a, 0x02, 0x00], F32STORE = [0x38, 0x02, 0x00];
    const VADD = [0xfd, 0xe4, 0x01], VSUB = [0xfd, 0xe5, 0x01], VMUL = [0xfd, 0xe6, 0x01], SPLAT = [0xfd, 0x13], LANE = (k) => [0xfd, 0x1f, k], FADD = [0x92];
    const VLOADO = (o) => [0xfd, 0x00, 0x00, o], VSTOREO = (o) => [0xfd, 0x0b, 0x00, o];   // memarg offset o (< 128: one byte)
    const X16LO = [0xfd, 0x87, 0x01], X16HI = [0xfd, 0x88, 0x01], X32LO = [0xfd, 0xa7, 0x01], X32HI = [0xfd, 0xa8, 0x01], CVT = [0xfd, 0xfa, 0x01];   // int8 -> int16 -> int32 -> float32, low and high halves
    const BLOCK = [0x02, 0x40], LOOP = [0x03, 0x40], END = [0x0b], BR = (i) => [0x0c, i], BRIF = (i) => [0x0d, i];
    const hsum = (acc) => [...GET(acc), ...LANE(0), ...GET(acc), ...LANE(1), ...FADD, ...GET(acc), ...LANE(2), ...FADD, ...GET(acc), ...LANE(3), ...FADD];
    function build() {
      const i32 = 0x7f, v128 = 0x7b;
      const types = section(1, vec([[0x60, ...vec(new Array(6).fill([i32])), 0x00], [0x60, ...vec(new Array(5).fill([i32])), 0x00], [0x60, ...vec(new Array(7).fill([i32])), 0x00]]));
      const imp = section(2, vec([[...str('env'), ...str('mem'), 0x02, 0x00, 0x01]]));
      const funcs = section(3, vec([[0], [1], [1], [1], [1], [0], [1], [2]]));
      const exp = section(7, vec([[...str('gauss'), 0x00, 0], [...str('matvec'), 0x00, 1], [...str('wsum'), 0x00, 2], [...str('matvec8'), 0x00, 3], [...str('wsum8'), 0x00, 4], [...str('gauss16'), 0x00, 5], [...str('matvec16'), 0x00, 6], [...str('gauss8'), 0x00, 7]]));
      const locals = [[5, i32], [2, v128]];                   // r j rb po Db acc diff -> five i32 then two v128
      const dotLoop = (np, loadA, loadB, accumulate) => [
        ...I32C(0), ...SET(np + 1), ...BLOCK, ...LOOP, ...GET(np + 1), ...GET(np + 4), ...GEU, ...BRIF(1),
        ...accumulate, ...GET(np + 1), ...I32C(16), ...ADD, ...SET(np + 1), ...BR(0), ...END, ...END];
      // gauss: params u c a R D out = 0..5; locals r=6 j=7 rb=8 po=9 Db=10 acc=11 diff=12
      const g = (() => {
        const np = 6, r = 6, j = 7, rb = 8, po = 9, Db = 10, acc = 11, diff = 12;
        const acc_ = [
          ...GET(0), ...GET(j), ...ADD, ...VLOAD, ...GET(1), ...GET(rb), ...ADD, ...GET(j), ...ADD, ...VLOAD, ...VSUB, ...SET(diff),
          ...GET(acc), ...GET(2), ...GET(rb), ...ADD, ...GET(j), ...ADD, ...VLOAD, ...GET(diff), ...VMUL, ...GET(diff), ...VMUL, ...VADD, ...SET(acc)];
        const row = [...VZERO, ...SET(acc), ...dotLoop(np, 0, 0, acc_), ...GET(po), ...hsum(acc), ...F32STORE];
        return [...GET(4), ...I32C(4), ...MUL, ...SET(Db), ...GET(5), ...SET(po), ...I32C(0), ...SET(r), ...I32C(0), ...SET(rb),
          ...BLOCK, ...LOOP, ...GET(r), ...GET(3), ...GEU, ...BRIF(1), ...row,
          ...GET(r), ...I32C(1), ...ADD, ...SET(r), ...GET(rb), ...GET(Db), ...ADD, ...SET(rb), ...GET(po), ...I32C(4), ...ADD, ...SET(po), ...BR(0), ...END, ...END, ...END];
      })();
      // matvec: params M v R D out = 0..4; locals r=5 j=6 rb=7 po=8 Db=9 acc=10 diff=11
      const m = (() => {
        const np = 5, r = 5, j = 6, rb = 7, po = 8, Db = 9, acc = 10;
        const acc_ = [...GET(acc), ...GET(0), ...GET(rb), ...ADD, ...GET(j), ...ADD, ...VLOAD, ...GET(1), ...GET(j), ...ADD, ...VLOAD, ...VMUL, ...VADD, ...SET(acc)];
        const row = [...VZERO, ...SET(acc), ...dotLoop(np, 0, 0, acc_), ...GET(po), ...hsum(acc), ...F32STORE];
        return [...GET(3), ...I32C(4), ...MUL, ...SET(Db), ...GET(4), ...SET(po), ...I32C(0), ...SET(r), ...I32C(0), ...SET(rb),
          ...BLOCK, ...LOOP, ...GET(r), ...GET(2), ...GEU, ...BRIF(1), ...row,
          ...GET(r), ...I32C(1), ...ADD, ...SET(r), ...GET(rb), ...GET(Db), ...ADD, ...SET(rb), ...GET(po), ...I32C(4), ...ADD, ...SET(po), ...BR(0), ...END, ...END, ...END];
      })();
      // wsum: params M s R D out = 0..4; locals r=5 j=6 rb=7 (po=8 unused: the strength pointer) Db=9 acc=10 (the splatted strength) diff=11
      const w = (() => {
        const np = 5, r = 5, j = 6, rb = 7, po = 8, Db = 9, acc = 10;
        const zero = [...I32C(0), ...SET(j), ...BLOCK, ...LOOP, ...GET(j), ...GET(Db), ...GEU, ...BRIF(1), ...GET(4), ...GET(j), ...ADD, ...VZERO, ...VSTORE, ...GET(j), ...I32C(16), ...ADD, ...SET(j), ...BR(0), ...END, ...END];
        const upd = [...GET(4), ...GET(j), ...ADD, ...GET(4), ...GET(j), ...ADD, ...VLOAD, ...GET(acc), ...GET(0), ...GET(rb), ...ADD, ...GET(j), ...ADD, ...VLOAD, ...VMUL, ...VADD, ...VSTORE];
        const row = [...GET(po), ...F32LOAD, ...SPLAT, ...SET(acc), ...dotLoop(np, 0, 0, upd)];
        return [...GET(3), ...I32C(4), ...MUL, ...SET(Db), ...zero, ...GET(1), ...SET(po), ...I32C(0), ...SET(r), ...I32C(0), ...SET(rb),
          ...BLOCK, ...LOOP, ...GET(r), ...GET(2), ...GEU, ...BRIF(1), ...row,
          ...GET(r), ...I32C(1), ...ADD, ...SET(r), ...GET(rb), ...GET(Db), ...ADD, ...SET(rb), ...GET(po), ...I32C(4), ...ADD, ...SET(po), ...BR(0), ...END, ...END, ...END];
      })();
      // int8 weights: sixteen at a time, widened to four float32 vectors (the per-row scale is the caller's: folded into the strengths for wsum8, applied to the dots for matvec8)
      // The dot-product kernels keep four independent running sums (one per vector of a sixteen-element step): a single sum makes every add wait for the one before it.
      const hsum4 = (acc) => [...GET(acc), ...GET(acc + 1), ...VADD, ...GET(acc + 2), ...GET(acc + 3), ...VADD, ...VADD, ...SET(acc), ...hsum(acc)];
      const zero4 = (acc) => [0, 1, 2, 3].flatMap((k) => [...VZERO, ...SET(acc + k)]);
      const rowsLoop = (r, rb, po, Rp, outP, stride, row) => [...GET(outP), ...SET(po), ...I32C(0), ...SET(r), ...I32C(0), ...SET(rb),
        ...BLOCK, ...LOOP, ...GET(r), ...GET(Rp), ...GEU, ...BRIF(1), ...row,
        ...GET(r), ...I32C(1), ...ADD, ...SET(r), ...GET(rb), ...stride, ...ADD, ...SET(rb), ...GET(po), ...I32C(4), ...ADD, ...SET(po), ...BR(0), ...END, ...END, ...END];
      const jLoop = (j, bound, step, inner) => [...I32C(0), ...SET(j), ...BLOCK, ...LOOP, ...GET(j), ...bound, ...GEU, ...BRIF(1), ...inner, ...GET(j), ...I32C(step), ...ADD, ...SET(j), ...BR(0), ...END, ...END];
      // gauss16(u, c, a, R, D, out): gauss for D a multiple of 16; params 0..5; locals r=6 j=7 (byte offset) rb=8 po=9 Db=10, v128 acc 11..14, diff 15
      const g16 = (() => {
        const r = 6, j = 7, rb = 8, po = 9, Db = 10, acc = 11, diff = 15;
        const one = (k) => [...GET(0), ...GET(j), ...ADD, ...VLOADO(16 * k), ...GET(1), ...GET(rb), ...ADD, ...GET(j), ...ADD, ...VLOADO(16 * k), ...VSUB, ...SET(diff),
          ...GET(acc + k), ...GET(2), ...GET(rb), ...ADD, ...GET(j), ...ADD, ...VLOADO(16 * k), ...GET(diff), ...VMUL, ...GET(diff), ...VMUL, ...VADD, ...SET(acc + k)];
        const row = [...zero4(acc), ...jLoop(j, GET(Db), 64, [0, 1, 2, 3].flatMap(one)), ...GET(po), ...hsum4(acc), ...F32STORE];
        return [...GET(4), ...I32C(4), ...MUL, ...SET(Db), ...rowsLoop(r, rb, po, 3, 5, GET(Db), row)];
      })();
      // matvec16(M, v, R, D, out): matvec for D a multiple of 16; params 0..4; locals r=5 j=6 rb=7 po=8 Db=9, v128 acc 10..13
      const m16 = (() => {
        const r = 5, j = 6, rb = 7, po = 8, Db = 9, acc = 10;
        const one = (k) => [...GET(acc + k), ...GET(0), ...GET(rb), ...ADD, ...GET(j), ...ADD, ...VLOADO(16 * k), ...GET(1), ...GET(j), ...ADD, ...VLOADO(16 * k), ...VMUL, ...VADD, ...SET(acc + k)];
        const row = [...zero4(acc), ...jLoop(j, GET(Db), 64, [0, 1, 2, 3].flatMap(one)), ...GET(po), ...hsum4(acc), ...F32STORE];
        return [...GET(3), ...I32C(4), ...MUL, ...SET(Db), ...rowsLoop(r, rb, po, 2, 4, GET(Db), row)];
      })();
      // matvec8(Mq, v, R, D, out): out[r] = sum_j q[r,j] v[j] (D a multiple of 16); locals r=5 j=6 (byte in the int8 row) rb=7 po=8 jf=9 (byte in v), v128 acc 10..13, x=14 y=15
      const m8 = (() => {
        const r = 5, j = 6, rb = 7, po = 8, jf = 9, acc = 10, x = 14, y = 15;
        const grp = (k, ext, src) => [...GET(src), ...ext, ...CVT, ...GET(1), ...GET(jf), ...ADD, ...VLOADO(16 * k), ...VMUL, ...GET(acc + k), ...VADD, ...SET(acc + k)];
        const inner = [...GET(0), ...GET(rb), ...ADD, ...GET(j), ...ADD, ...VLOAD, ...SET(x),
          ...GET(x), ...X16HI, ...SET(y), ...GET(x), ...X16LO, ...SET(x),
          ...grp(0, X32LO, x), ...grp(1, X32HI, x), ...grp(2, X32LO, y), ...grp(3, X32HI, y), ...GET(jf), ...I32C(64), ...ADD, ...SET(jf)];
        const row = [...zero4(acc), ...I32C(0), ...SET(jf), ...jLoop(j, GET(3), 16, inner), ...GET(po), ...hsum4(acc), ...F32STORE];
        return [...rowsLoop(r, rb, po, 2, 4, GET(3), row)];
      })();
      // wsum8(Mq, s, R, D, out): out[j] = sum_r s[r] q[r,j]       (D a multiple of 16); locals r=5 j=6 rb=7 po=8 (strength pointer) jf=9, v128 sp=10 x=11 y=12
      const w8 = (() => {
        const r = 5, j = 6, rb = 7, po = 8, jf = 9, sp = 10, x = 11, y = 12;
        const zero = [...I32C(0), ...SET(jf), ...BLOCK, ...LOOP, ...GET(jf), ...GET(3), ...I32C(4), ...MUL, ...GEU, ...BRIF(1), ...GET(4), ...GET(jf), ...ADD, ...VZERO, ...VSTORE, ...GET(jf), ...I32C(16), ...ADD, ...SET(jf), ...BR(0), ...END, ...END];
        const grp = (ext, src, off) => [...GET(4), ...GET(jf), ...ADD, ...GET(4), ...GET(jf), ...ADD, ...VLOADO(off), ...GET(src), ...ext, ...CVT, ...GET(sp), ...VMUL, ...VADD, ...VSTOREO(off)];
        const inner = [
          ...GET(0), ...GET(rb), ...ADD, ...GET(j), ...ADD, ...VLOAD, ...SET(x),
          ...GET(x), ...X16HI, ...SET(y), ...GET(x), ...X16LO, ...SET(x),
          ...grp(X32LO, x, 0), ...grp(X32HI, x, 16), ...grp(X32LO, y, 32), ...grp(X32HI, y, 48)];
        const row = [...GET(po), ...F32LOAD, ...SPLAT, ...SET(sp), ...I32C(0), ...SET(j), ...I32C(0), ...SET(jf), ...BLOCK, ...LOOP, ...GET(j), ...GET(3), ...GEU, ...BRIF(1),
          ...inner, ...GET(j), ...I32C(16), ...ADD, ...SET(j), ...GET(jf), ...I32C(64), ...ADD, ...SET(jf), ...BR(0), ...END, ...END];
        return [...zero, ...GET(1), ...SET(po), ...I32C(0), ...SET(r), ...I32C(0), ...SET(rb),
          ...BLOCK, ...LOOP, ...GET(r), ...GET(2), ...GEU, ...BRIF(1), ...row,
          ...GET(r), ...I32C(1), ...ADD, ...SET(r), ...GET(rb), ...GET(3), ...ADD, ...SET(rb), ...GET(po), ...I32C(4), ...ADD, ...SET(po), ...BR(0), ...END, ...END, ...END];
      })();
      // gauss8(u, cq, cs, a, R, D, out): gauss with int8 centres (per-row scale at cs) and float widths; params 0..6; locals r=7 j=8 (byte in the int8 row) jf=9 (byte in a float row) rq=10 rf=11 po=12 sp=13, v128 acc 14..17 x=18 y=19 sc=20 diff=21
      const g8 = (() => {
        const r = 7, j = 8, jf = 9, rq = 10, rf = 11, po = 12, sp = 13, acc = 14, x = 18, y = 19, sc = 20, diff = 21;
        const one = (k, ext, src) => [...GET(0), ...GET(jf), ...ADD, ...VLOADO(16 * k), ...GET(src), ...ext, ...CVT, ...GET(sc), ...VMUL, ...VSUB, ...SET(diff),
          ...GET(acc + k), ...GET(3), ...GET(rf), ...ADD, ...GET(jf), ...ADD, ...VLOADO(16 * k), ...GET(diff), ...VMUL, ...GET(diff), ...VMUL, ...VADD, ...SET(acc + k)];
        const inner = [...GET(1), ...GET(rq), ...ADD, ...GET(j), ...ADD, ...VLOAD, ...SET(x),
          ...GET(x), ...X16HI, ...SET(y), ...GET(x), ...X16LO, ...SET(x),
          ...one(0, X32LO, x), ...one(1, X32HI, x), ...one(2, X32LO, y), ...one(3, X32HI, y), ...GET(jf), ...I32C(64), ...ADD, ...SET(jf)];
        const row = [...GET(sp), ...F32LOAD, ...SPLAT, ...SET(sc), ...zero4(acc), ...I32C(0), ...SET(jf), ...jLoop(j, GET(5), 16, inner), ...GET(po), ...hsum4(acc), ...F32STORE];
        return [...GET(6), ...SET(po), ...GET(2), ...SET(sp), ...I32C(0), ...SET(r), ...I32C(0), ...SET(rq), ...I32C(0), ...SET(rf),
          ...BLOCK, ...LOOP, ...GET(r), ...GET(4), ...GEU, ...BRIF(1), ...row,
          ...GET(r), ...I32C(1), ...ADD, ...SET(r), ...GET(rq), ...GET(5), ...ADD, ...SET(rq), ...GET(rf), ...GET(5), ...I32C(4), ...MUL, ...ADD, ...SET(rf), ...GET(po), ...I32C(4), ...ADD, ...SET(po), ...GET(sp), ...I32C(4), ...ADD, ...SET(sp), ...BR(0), ...END, ...END, ...END];
      })();
      const localsG8 = [[7, i32], [8, v128]];
      const locals8 = [[5, i32], [3, v128]], localsM8 = [[5, i32], [6, v128]], localsG16 = [[5, i32], [5, v128]], localsM16 = [[5, i32], [4, v128]];
      const body = (code, loc) => { const b = [...vec(loc || locals), ...code]; return [...leb(b.length), ...b]; };
      const codes = section(10, vec([body(g), body(m), body(w), body(m8, localsM8), body(w8, locals8), body(g16, localsG16), body(m16, localsM16), body(g8, localsG8)]));
      return new Uint8Array([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, ...types, ...imp, ...funcs, ...exp, ...codes]);
    }
    let cached;
    function make(memory) {
      try {
        if (typeof WebAssembly !== 'object') return null;
        const bytes = build();
        if (!WebAssembly.validate(bytes)) return null;
        const inst = new WebAssembly.Instance(new WebAssembly.Module(bytes), { env: { mem: memory } });
        return inst.exports;
      } catch (e) { return null; }
    }
    return { make, build, ok: () => { if (cached === undefined) { try { cached = typeof WebAssembly === 'object' && WebAssembly.validate(build()); } catch (e) { cached = false; } } return cached; } };
  })();


  function initKernels(m) {                                // the big weight arrays move into one WebAssembly memory (the model's own arrays become views of it, so in-place edits such as surgery still work)
    if (!Wasm.ok()) return null;
    const { V, d, L, W } = m;
    let floats = m.embH.length + 16 * d + 8 * V + 64, q8 = FLM.int8Kernels !== false && d % 16 === 0;   // q8: the slopes and consequents stay int8 in the kernels' memory (a quarter of the bytes to stream)
    for (let l = 0; l < L; l++) {
      const fc = W['blocks.' + l + '.ffn.centers'], R = fc.shape[0], D = fc.shape[1];
      if (D % 4 || d % 4) return null;
      if (!(W['blocks.' + l + '.ffn.slope.weight'].q && W['blocks.' + l + '.ffn.conseq'].q)) q8 = false;
      floats += 2 * R * D + 2 * R * d + 8 * R + 4 * D + (R * d) / 2;
    }
    const pages = Math.ceil(((floats + 4096) * 4) / 65536) + 1;
    let mem; try { mem = new WebAssembly.Memory({ initial: pages }); } catch (e) { return null; }
    const ex = Wasm.make(mem); if (!ex) return null;
    let off = 0;
    const alloc = (n, src) => { const o = off; off += (n + 3) & ~3; const v = new Float32Array(mem.buffer, o * 4, n); if (src) v.set(src); return { o: o * 4, v }; };
    const allocB = (n, src) => { const o = off; off += ((n + 15) >> 4) << 2; const v = new Int8Array(mem.buffer, o * 4, n); if (src) v.set(src); return { o: o * 4, v }; };
    const Eh = alloc(m.embH.length, m.embH);               // the crisp head rows only: the tail's scores are built from the prototypes (see step)
    const layers = [];
    for (let l = 0; l < L; l++) {
      const b = 'blocks.' + l + '.ffn.', fc = W[b + 'centers'], R = fc.shape[0], D = fc.shape[1];
      const c = alloc(R * D, fc.data), a = alloc(R * D, m.A[l].ffn);
      fc.data = c.v; m.A[l].ffn = a.v;
      if (q8) { const sq = W[b + 'slope.weight'], cqq = W[b + 'conseq']; layers.push({ c, sw: allocB(R * d, sq.q), cq: allocB(R * d, cqq.q), swS: sq.scale, cqS: cqq.scale, a, R, D }); continue; }
      const sw = alloc(R * d, W[b + 'slope.weight'].data), cq = alloc(R * d, W[b + 'conseq'].data);
      W[b + 'slope.weight'].data = sw.v; W[b + 'conseq'].data = cq.v;
      layers.push({ c, sw, cq, a, R, D });
    }
    const maxR = Math.max(...layers.map((q) => q.R)), maxD = Math.max(...layers.map((q) => q.D));
    const buf = { u: alloc(maxD), n2: alloc(d), fl: alloc(maxR), dots: alloc(maxR), st: alloc(maxR), dx: alloc(d), go: alloc(d), logits: alloc(V) };
    return { ex, mem, Eh, layers, buf, q8 };
  }

  class AllFuzzyModel {
    constructor(manifest, buf) {
      const c = manifest.config;
      this.manifest = manifest; this.cfg = c;
      const W = (this.W = readTensors(manifest, buf));
      const g = (k) => W[k].data;
      this.V = c.vocab_size; this.d = c.d_model; this.H = c.n_heads; this.dh = this.d / this.H; this.R = c.n_rules; this.L = c.n_layers; this.maxLen = c.max_len;
      // c.tanh: signed (tanh) memberships for the tail table; a fuzzy head table arrives as precomputed effective rows in emb_h.weight
      const headIds = g('head_ids'), tailIds = g('tail_ids'), embH = g('emb_h.weight'), embT = g('emb_t.weight'), up = g('tier_up.weight'), K = W['emb_t.weight'].shape[1], d = this.d;
      this.E = new Float32Array(this.V * d);              // effective embedding of every piece: crisp row, or membership-weighted average of the 12 prototypes
      headIds.forEach((id, i) => this.E.set(embH.subarray(i * d, i * d + d), id * d));
      for (let t = 0; t < tailIds.length; t++) {
        const lg = embT.subarray(t * K, t * K + K), mu = c.tanh ? lg.map(Math.tanh) : softmaxF(lg), id = tailIds[t];
        for (let j = 0; j < d; j++) { let a = 0; for (let k = 0; k < K; k++) a += up[j * K + k] * mu[k]; this.E[id * d + j] = a; }
      }
      // output scores without the full table: a head row's score is its own dot product; a tail piece's row is up . mu (its memberships over the K prototypes), so its score is mu . (up^T go): K dot products once per step
      this.embH = embH; this.headIds = headIds; this.tailIds = tailIds; this.up = up; this.KP = K;
      this.tailMu = new Float32Array(tailIds.length * K);
      for (let t = 0; t < tailIds.length; t++) { const lg = embT.subarray(t * K, t * K + K), mu = c.tanh ? lg.map(Math.tanh) : softmaxF(lg); this.tailMu.set(mu, t * K); }
      this.nw = g('norm.weight');
      const wd = (k) => { const x = g(k), o = x.length > 65536 ? new Float32Array(x.length) : new Float64Array(x.length); for (let i = 0; i < x.length; i++) o[i] = Math.exp(-2 * x[i]); return o; };   // large width arrays in float32: half the memory traffic
      this.A = [];                                         // 1 / s^2 of every membership function
      for (let l = 0; l < this.L; l++) { const b = 'blocks.' + l + '.'; this.A.push({ mem: wd(b + 'mix.log_width'), qkv: wd(b + 'mix.qkv.log_width'), out: wd(b + 'mix.out.log_width'), gate: wd(b + 'mix.gate.log_width'), beta: wd(b + 'mix.beta.log_width'), ffn: wd(b + 'ffn.log_width') }); }
      this.pi = null;
      this.K = initKernels(this);
      if (this.K && this.K.q8) this.requantRow = (l, r) => {            // surgery edited one consequent row (float32 in W): rebuild its int8 row and scale
        const Kl = this.K.layers[l], d = this.d, cq = this.W['blocks.' + l + '.ffn.conseq'].data;
        let mx = 0; for (let i = 0; i < d; i++) mx = Math.max(mx, Math.abs(cq[r * d + i]));
        const sc = Math.max(mx, 1e-12) / 127; Kl.cqS[r] = sc;
        const q = Kl.cq.v; for (let i = 0; i < d; i++) q[r * d + i] = Math.max(-127, Math.min(127, Math.round(cq[r * d + i] / sc)));
      };                           // null where WebAssembly SIMD is missing: the plain loops below then do the work
      this.reset();
    }
    reset() {
      this.S = Array.from({ length: this.L }, () => new Float64Array(this.H * this.dh * this.R));
      this.prevN2 = Array.from({ length: this.L }, () => new Float64Array(this.d));
      this.pos = 0;
    }
    gain(x, p) {                                           // TSK gain controller: one input variable, z = log rms(x), five rules, defuzzified log-gain
      const W = this.W, c = W[p + 'centers'].data, lw = W[p + 'log_width'].data, off = W[p + 'offset'].data, sl = W[p + 'slope'].data, wt = W[p + 'weight'].data;
      let ss = 0; for (let i = 0; i < x.length; i++) ss += x[i] * x[i];
      const z = 0.5 * Math.log(ss / x.length + EPS), lg = new Float64Array(c.length);
      for (let r = 0; r < c.length; r++) { const t = (z - c[r]) / Math.exp(lw[r]); lg[r] = -0.5 * t * t; }
      const w = softmaxF(lg); let gn = 0; for (let r = 0; r < c.length; r++) gn += w[r] * (off[r] + sl[r] * (z - c[r]));
      const e = Math.exp(gn), out = new Float64Array(x.length); for (let i = 0; i < x.length; i++) out[i] = x[i] * e * wt[i];
      return { out, gain: gn };
    }
    gatedMap(x, p, a, outDim) {                            // rule-gated low-rank map: u = B x, rules over u rescale the basis directions, y = A (g * u)
      const W = this.W, B = W[p + 'B.weight'], Am = W[p + 'A.weight'].data, c = W[p + 'centers'], s = W[p + 's'].data, k = B.shape[0], d = x.length, R = c.shape[0];
      const u = new Float64Array(k); for (let o = 0; o < k; o++) { let v = 0; for (let i = 0; i < d; i++) v += B.data[o * d + i] * x[i]; u[o] = v; }
      const w = softmaxF(gaussLogits(u, c.data, a, R, k, 1 / k)), gv = new Float64Array(k);
      for (let j = 0; j < k; j++) { let v = 0; for (let r = 0; r < R; r++) v += w[r] * s[r * k + j]; gv[j] = v * u[j]; }
      const y = new Float64Array(outDim); for (let o = 0; o < outDim; o++) { let v = 0; for (let j = 0; j < k; j++) v += Am[o * k + j] * gv[j]; y[o] = v; }
      return y;
    }
    ruleGate(x, p, a) {                                    // six Gaussian rules over the whole input; consequents are per-head logits; weighted average
      const W = this.W, c = W[p + 'centers'], cs = W[p + 'conseq'].data, R = c.shape[0], H = this.H;
      const w = softmaxF(gaussLogits(x, c.data, a, R, x.length, 1 / x.length)), out = new Float64Array(H);
      for (let h = 0; h < H; h++) { let v = 0; for (let r = 0; r < R; r++) v += w[r] * cs[r * H + h]; out[h] = v; }
      return out;
    }
    step(tok, opts) {
      const { d, H, dh, R, W } = this;
      const x = Float64Array.from(this.E.subarray(tok * d, tok * d + d));
      const strengths = [];
      for (let l = 0; l < this.L; l++) {
        const b = 'blocks.' + l + '.', A = this.A[l];
        const h = this.gain(x, b + 'n1.').out;
        const qkv = this.gatedMap(h, b + 'mix.qkv.', A.qkv, 3 * d);
        const al = this.ruleGate(h, b + 'mix.gate.', A.gate), be = this.ruleGate(h, b + 'mix.beta.', A.beta);
        const mc = W[b + 'mix.centers'].data, y = new Float64Array(d), S = this.S[l];
        for (let hd = 0; hd < H; hd++) {
          const co = hd * R * dh, cs = mc.subarray(co, co + R * dh), as = A.mem.subarray(co, co + R * dh);
          const fq = softmaxF(gaussLogits(qkv.subarray(hd * dh, hd * dh + dh), cs, as, R, dh, 1 / dh));
          const fk = softmaxF(gaussLogits(qkv.subarray(d + hd * dh, d + hd * dh + dh), cs, as, R, dh, 1 / dh));
          const alpha = sigmoid(al[hd]), beta = sigmoid(be[hd]), base = hd * dh * R;
          for (let j = 0; j < dh; j++) {                  // forget, predict, write the error into the rules fk selects, read with fq
            const row = base + j * R, vj = qkv[2 * d + hd * dh + j]; let pred = 0;
            for (let r = 0; r < R; r++) { S[row + r] *= alpha; pred += S[row + r] * fk[r]; }
            const e = beta * (vj - pred); let yo = 0;
            for (let r = 0; r < R; r++) { S[row + r] += e * fk[r]; yo += S[row + r] * fq[r]; }
            y[hd * dh + j] = yo;
          }
        }
        const mo = this.gatedMap(y, b + 'mix.out.', A.out, d); for (let i = 0; i < d; i++) x[i] += mo[i];
        const n2 = this.gain(x, b + 'n2.').out;
        const fc = W[b + 'ffn.centers'], D = fc.shape[1], FR = fc.shape[0];
        let u = n2; if (D === 2 * d) { u = new Float64Array(D); u.set(n2, 0); u.set(this.prevN2[l], d); }
        const K = this.K, sb = W[b + 'ffn.slope.bias'].data, st = new Float64Array(FR);
        if (K) {                                                // squared distances, slopes and the weighted consequents: four lanes at a time, float32
          const Kl = K.layers[l], B = K.buf;
          B.u.v.set(u); B.n2.v.set(n2);
          (D % 16 ? K.ex.gauss : K.ex.gauss16)(B.u.o, Kl.c.o, Kl.a.o, FR, D, B.fl.o);
          const q = B.fl.v, dt = B.dots.v, sv = B.st.v, c0 = -0.5 / D;
          if (K.q8) {                                           // int8 slopes and consequents: the row scales are folded into the dots and the strengths
            K.ex.matvec8(Kl.sw.o, B.n2.o, FR, d, B.dots.o);
            const ss = Kl.swS, cs = Kl.cqS;
            for (let r = 0; r < FR; r++) { const s_ = Math.exp(c0 * q[r]) * (sb[r] + dt[r] * ss[r]); st[r] = s_; sv[r] = s_ * cs[r]; }
            K.ex.wsum8(Kl.cq.o, B.st.o, FR, d, B.dx.o);
          } else {
            (d % 16 ? K.ex.matvec : K.ex.matvec16)(Kl.sw.o, B.n2.o, FR, d, B.dots.o);
            for (let r = 0; r < FR; r++) { const s_ = Math.exp(c0 * q[r]) * (sb[r] + dt[r]); st[r] = s_; sv[r] = s_; }
            K.ex.wsum(Kl.cq.o, B.st.o, FR, d, B.dx.o);
          }
          const dx = B.dx.v; for (let i = 0; i < d; i++) x[i] += dx[i];
          this.prevN2[l] = n2; strengths.push(st);
          continue;
        }
        const fl = gaussLogits(u, fc.data, A.ffn, FR, D, 1 / D), sw = W[b + 'ffn.slope.weight'].data, cq = W[b + 'ffn.conseq'].data;
        for (let r = 0; r < FR; r++) {
          let sl = sb[r]; const o = r * d; for (let i = 0; i < d; i++) sl += sw[o + i] * n2[i];
          const s_ = Math.exp(fl[r]) * sl; st[r] = s_;
          for (let i = 0; i < d; i++) x[i] += s_ * cq[o + i];
        }
        this.prevN2[l] = n2; strengths.push(st);
      }
      const gf = this.gain(x, 'norm.'), logits = new Float32Array(this.V);
      const go = gf.out, nh = this.headIds.length, KP = this.KP, up = this.up, tailIds = this.tailIds, tmu = this.tailMu;
      let hd;
      if (this.K) { const B = this.K.buf; B.go.v.set(go); (d % 16 ? this.K.ex.matvec : this.K.ex.matvec16)(this.K.Eh.o, B.go.o, nh, d, B.logits.o); hd = B.logits.v; }
      else {
        hd = new Float32Array(nh); const eh = this.embH;
        for (let i = 0; i < nh; i++) { let a0 = 0, a1 = 0, a2 = 0, a3 = 0, j = 0; const r = i * d; for (; j < d - (d % 4); j += 4) { a0 += eh[r + j] * go[j]; a1 += eh[r + j + 1] * go[j + 1]; a2 += eh[r + j + 2] * go[j + 2]; a3 += eh[r + j + 3] * go[j + 3]; } for (; j < d; j++) a0 += eh[r + j] * go[j]; hd[i] = a0 + a1 + a2 + a3; }
      }
      for (let i = 0; i < nh; i++) logits[this.headIds[i]] = hd[i];
      const z = new Float64Array(KP); for (let j = 0; j < d; j++) { const gj = go[j], o = j * KP; for (let k = 0; k < KP; k++) z[k] += up[o + k] * gj; }
      for (let t = 0; t < tailIds.length; t++) { let a = 0; const o = t * KP; for (let k = 0; k < KP; k++) a += tmu[o + k] * z[k]; logits[tailIds[t]] = a; }
      this.pos++;
      return opts && opts.explain ? { logits, gates: strengths, s: Math.exp(gf.gain), xPre: Float64Array.from(x) } : { logits };
    }
    profileRow(l, r) {                                      // pi_r over all pieces, computed when first asked for and kept
      this.rows = this.rows || {};
      const key = l * 1000003 + r;
      if (!this.rows[key]) {
        const { d, V, E } = this, cq = this.W['blocks.' + l + '.ffn.conseq'].data, cw = new Float32Array(d), pi = new Float32Array(V);
        for (let i = 0; i < d; i++) cw[i] = cq[r * d + i] * this.nw[i];
        for (let v = 0; v < V; v++) { let a = 0; const o = v * d; for (let i = 0; i < d; i++) a += cw[i] * E[o + i]; pi[v] = a; }
        this.rows[key] = pi;
      }
      return this.rows[key];
    }
    profileAt(l, tok) {                                     // pi_r[tok] for every rule r of layer l: one dot product per rule
      const { d, E, nw } = this, cq = this.W['blocks.' + l + '.ffn.conseq'], R = cq.shape[0], out = new Float32Array(R), o = tok * d, ew = new Float32Array(d);
      for (let i = 0; i < d; i++) ew[i] = E[o + i] * nw[i];
      for (let r = 0; r < R; r++) { let a = 0; const b = r * d; for (let i = 0; i < d; i++) a += cq.data[b + i] * ew[i]; out[r] = a; }
      return out;
    }
    profiles() {                                           // pi_r = (conseq_r * output weight) . E^T: what rule r does to the output scores when it fires positively
      if (this.pi) return this.pi;
      const { d, V } = this;
      this.pi = Array.from({ length: this.L }, (_, l) => {
        const cq = this.W['blocks.' + l + '.ffn.conseq'], R = cq.shape[0], pi = new Float32Array(R * V), cw = new Float32Array(d);
        for (let r = 0; r < R; r++) {
          for (let i = 0; i < d; i++) cw[i] = cq.data[r * d + i] * this.nw[i];
          for (let v = 0; v < V; v++) { let a = 0; for (let i = 0; i < d; i++) a += cw[i] * this.E[v * d + i]; pi[r * V + v] = a; }
        }
        return pi;
      });
      return this.pi;
    }
    ruleShape(layer, rule) {                               // the rule's most selective membership functions, read from the weights
      const fc = this.W['blocks.' + layer + '.ffn.centers'], lw = this.W['blocks.' + layer + '.ffn.log_width'].data, D = fc.shape[1], dims = [];
      for (let j = 0; j < D; j++) dims.push({ dim: j, center: fc.data[rule * D + j], sigma: Math.exp(lw[rule * D + j]) * Math.sqrt(D) });
      return dims.sort((a, b) => a.sigma - b.sigma).slice(0, 24);
    }
    mixerRule(layer, head, rule) {
      const b = (head * this.R + rule) * this.dh, c = this.W['blocks.' + layer + '.mix.centers'].data, lw = this.W['blocks.' + layer + '.mix.log_width'].data, cc = [], w = [];
      for (let j = 0; j < this.dh; j++) { cc.push(c[b + j]); w.push(Math.exp(lw[b + j])); }
      return { layer, head, rule, center: cc, width: w };
    }
  }

  // ------------------------------------------------------------------ sampling
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function softmax(v, T) {
    let mx = -Infinity; for (const x of v) if (x > mx) mx = x;
    const p = new Float64Array(v.length); let s = 0;
    for (let i = 0; i < v.length; i++) { p[i] = Math.exp((v[i] - mx) / T); s += p[i]; }
    for (let i = 0; i < p.length; i++) p[i] /= s;
    return p;
  }
  function topIdx(a, k) {                                  // indices of the k largest entries, descending (ties: the lower index first, as a stable sort would)
    const n = a.length, best = [];
    if (k >= n) { const idx = Array.from({ length: n }, (_, i) => i); idx.sort((i, j) => a[j] - a[i]); return idx; }
    for (let i = 0; i < n; i++) {
      const v = a[i];
      if (best.length < k) best.push(i); else if (v > a[best[k - 1]]) best[k - 1] = i; else continue;
      for (let j = best.length - 1; j > 0 && a[best[j - 1]] < a[best[j]]; j--) { const t = best[j]; best[j] = best[j - 1]; best[j - 1] = t; }
    }
    return best;
  }
  const rnd = (x, n) => Math.round(x * 10 ** n) / 10 ** n;

  // ------------------------------------------------------------------ the demo API (same JSON shapes as demo/server.py)
  class Api {
    constructor(base, fetchJson, fetchBuf) {
      this.base = base || '.'; this.fj = fetchJson; this.fb = fetchBuf; this.loaded = {}; this.tokp = null; this.idx = null;
    }
    async models() { return (this.idx = this.idx || (await this.fj(this.base + '/models/index.json'))); }
    async tok(name) {                                      // one tokenizer per vocabulary: bpe1024 (tokenizer.json), bpe3072 (tokenizer-3072.json)
      const f = name && name !== 'bpe1024' ? '/tokenizer-' + name.replace('bpe', '') + '.json' : '/tokenizer.json';
      this.tokp = this.tokp || {};
      return (this.tokp[f] = this.tokp[f] || this.fj(this.base + f).then((j) => new Tokenizer(j)));
    }
    async load(id) {
      if (!this.loaded[id]) {
        this.loaded[id] = (async () => {
          const man = await this.fj(`${this.base}/models/${id}.json`);
          const [buf, rules] = await Promise.all([this.fb(`${this.base}/models/${id}.bin`), this.fj(`${this.base}/models/${id}.rules.json`)]);
          return { model: man.config.allfuzzy ? new AllFuzzyModel(man, buf) : new Model(man, buf), rules, man };
        })();
      }
      return this.loaded[id];
    }
    async structure(id) { return (await this.load(id)).rules.structure; }
    async rules(id) { return (await this.load(id)).rules.rules; }
    async rule(id, layer, rule) {
      const { model, rules } = await this.load(id), tok = await this.tok();
      const hit = rules.detail[layer + '.' + rule];
      if (hit) {
        const out = JSON.parse(JSON.stringify(hit));
        for (const dm of out.dims) { dm.all = dm.all.map((v) => v / 100); dm.fires = dm.fires.map((v) => v / 100); }
        return out;
      }
      // not precomputed (large model): membership functions from the weights, no data statistics
      const dims = model.ruleShape(layer, rule).map((m) => ({
        dim: m.dim, center: rnd(m.center, 3), sigma: rnd(m.sigma, 3), data_sd: 0, selectivity: 0,
        lo: rnd(m.center - 3 * m.sigma, 3), hi: rnd(m.center + 3 * m.sigma, 3), all: [], fires: [],
      }));
      return { layer, rule, firing_mean: null, firing_p99: null, dims, contexts: [], note: 'membership functions from the weights; data histograms are precomputed only for the most used rules of this model' };
    }
    async mixrule(id, layer, head, rule) {
      const { model } = await this.load(id);
      const m = model.mixerRule(layer, head, rule);
      return { ...m, center: m.center.map((v) => rnd(v, 3)), width: m.width.map((v) => rnd(v, 3)) };
    }
    async generate(req) {
      const { model, man } = await this.load(req.model), tok = await this.tok(man.tok);
      const prompt = req.prompt || 'Once upon a time';
      const ctx = model.maxLen, ids = tok.encode(prompt);
      if (!ids.length) throw new Error('empty prompt');
      if (ids.length >= ctx) throw new Error(`prompt is ${ids.length} word pieces; the context is ${ctx}`);
      const nNew = Math.max(1, Math.min(+req.max_tokens || 80, 400, ctx - ids.length));
      const temp = Math.max(0.05, +req.temperature || 0.5), topK = Math.max(1, +req.top_k || 10);
      const wantExplain = req.explain !== false, useDry = req.dry !== false;
      const rng = mulberry32(+req.seed || 0);
      model.reset();
      let last = null;
      for (let i = 0; i < ids.length; i++) last = model.step(ids[i], { explain: wantExplain && i === ids.length - 1 });
      const hist = ids.slice(), seen = new Set();
      for (let i = 0; i + 2 < hist.length; i++) seen.add(hist[i] + ',' + hist[i + 1] + ',' + hist[i + 2]);
      const lazy = wantExplain && !!model.profileRow, steps = [], pi = wantExplain && !lazy ? model.profiles() : null;
      for (let n = 0; n < nNew; n++) {
        const logits = last.logits, pAll = softmax(logits, 1);
        let lgs = logits;   // the scores the sampler sees: the raw scores minus the DRY penalties (the displayed probabilities stay raw)
        if (useDry) { lgs = Array.from(logits); for (const [t, p] of dryPenalties(hist)) lgs[t] -= p; }
        const order = topIdx(lgs, Math.min(topK, lgs.length)), kth = lgs[order[order.length - 1]];
        const lg = Array.from(lgs, (v) => (v < kth ? -Infinity : v));
        const probs = softmax(lg, temp);
        let u = rng(), tk = order[0], acc = 0;
        for (let i = 0; i < probs.length; i++) { acc += probs[i]; if (u < acc) { tk = i; break; } }
        const step = {
          text: tok.piece(tk), p: rnd(pAll[tk], 4), id: tk, topIds: topIdx(pAll, 5),
          top: topIdx(pAll, 5).map((i) => ({ text: tok.piece(i), p: rnd(pAll[i], 4) })),
        };
        if (wantExplain && last.gates) {
          step.rules = []; step.gates = [];
          last.gates.forEach((g, l) => {
            step.gates.push(Array.from(g, (v) => rnd(v, 2)));
            const col = lazy ? model.profileAt(l, tk) : null;
            const contrib = Array.from(g, (gr, r) => gr * last.s * (lazy ? col[r] : pi[l][r * model.V + tk]));
            for (const r of topIdx(contrib.map(Math.abs), 3)) {
              const sign = g[r] >= 0 ? 1 : -1, row = lazy ? model.profileRow(l, r) : null, prof = Array.from({ length: model.V }, (_, v) => (lazy ? row[v] : pi[l][r * model.V + v]) * sign);
              step.rules.push({ layer: l, rule: r, gate: rnd(g[r], 3), contribution: rnd(contrib[r], 3), raises: topIdx(prof, 5).map((v) => tok.piece(v)) });
            }
          });
        }
        if (wantExplain && last.gates && last.xPre) step.surg = { x: last.xPre, g: last.gates[last.gates.length - 1] };   // for FLM.Surgery.plan
        const tri = hist.length >= 2 ? hist[hist.length - 2] + ',' + hist[hist.length - 1] + ',' + tk : null;
        step.rep = !!tri && seen.has(tri);
        if (tri) seen.add(tri);
        hist.push(tk); steps.push(step);
        if (n + 1 < nNew) last = model.step(tk, { explain: wantExplain });
        if ((n & 15) === 15) await new Promise((r) => setTimeout(r, 0));   // let the page repaint
      }
      return { prompt, steps, explained: wantExplain && steps.length > 0 && !!steps[0].rules };
    }
  }

  // DRY ("don't repeat yourself", PLAN_REPEAT.md set 25): a penalty on the token that would extend a sequence already repeated in the history.
  // For every earlier position j whose token equals the last token, the match length m is how far the history agrees backwards with the end of the text;
  // if m >= allowed the token that followed it (hist[j + 1]) is penalised by mult * base^(m - allowed). Same rule as flm/decode_sweep.py (S8: 0.8, 1.75, 2).
  function dryPenalties(hist, mult = 0.8, base = 1.75, allowed = 2, maxLen = 20) {
    const pen = new Map(), n = hist.length;
    if (n < 2) return pen;
    for (let j = 0; j < n - 1; j++) {
      if (hist[j] !== hist[n - 1]) continue;
      let m = 1;
      while (m < maxLen && j - m >= 0 && hist[j - m] === hist[n - 1 - m]) m++;
      if (m >= allowed) { const p = mult * Math.pow(base, m - allowed), t = hist[j + 1]; if (!(pen.get(t) >= p)) pen.set(t, p); }
    }
    return pen;
  }

  const FLM = { Tokenizer, Model, AllFuzzyModel, Api, mulberry32, softmax, dryPenalties };
  if (typeof module !== 'undefined' && module.exports) module.exports = FLM; else root.FLM = FLM;
})(typeof window !== 'undefined' ? window : globalThis);
