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
      for (let i = 0; i < n; i++) a[i] = fp16 ? half(dv.getUint16(t.offset + i * 2, true)) : dv.getFloat32(t.offset + i * 4, true);
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
    for (let r = 0; r < R; r++) { let q = 0; const b = r * D; for (let j = 0; j < D; j++) { const df = u[j] - c[b + j]; q += a[b + j] * df * df; } out[r] = -0.5 * scale * q; }
    return out;
  }
  function softmaxF(v) { let mx = -Infinity; for (const x of v) if (x > mx) mx = x; let s = 0; const o = new Float64Array(v.length); for (let i = 0; i < v.length; i++) { o[i] = Math.exp(v[i] - mx); s += o[i]; } for (let i = 0; i < o.length; i++) o[i] /= s; return o; }
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
      this.nw = g('norm.weight');
      const wd = (k) => { const x = g(k), o = new Float64Array(x.length); for (let i = 0; i < x.length; i++) o[i] = Math.exp(-2 * x[i]); return o; };
      this.A = [];                                         // 1 / s^2 of every membership function
      for (let l = 0; l < this.L; l++) { const b = 'blocks.' + l + '.'; this.A.push({ mem: wd(b + 'mix.log_width'), qkv: wd(b + 'mix.qkv.log_width'), out: wd(b + 'mix.out.log_width'), gate: wd(b + 'mix.gate.log_width'), beta: wd(b + 'mix.beta.log_width'), ffn: wd(b + 'ffn.log_width') }); }
      this.pi = null;
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
        const fl = gaussLogits(u, fc.data, A.ffn, FR, D, 1 / D), sw = W[b + 'ffn.slope.weight'].data, sb = W[b + 'ffn.slope.bias'].data, cq = W[b + 'ffn.conseq'].data;
        const st = new Float64Array(FR);
        for (let r = 0; r < FR; r++) { let sl = sb[r]; for (let i = 0; i < d; i++) sl += sw[r * d + i] * n2[i]; st[r] = Math.exp(fl[r]) * sl; }
        for (let r = 0; r < FR; r++) { const s_ = st[r]; for (let i = 0; i < d; i++) x[i] += s_ * cq[r * d + i]; }
        this.prevN2[l] = n2; strengths.push(st);
      }
      const gf = this.gain(x, 'norm.'), logits = new Float32Array(this.V);
      for (let v = 0; v < this.V; v++) { let a = 0; const r = v * d; for (let i = 0; i < d; i++) a += this.E[r + i] * gf.out[i]; logits[v] = a; }
      this.pos++;
      return opts && opts.explain ? { logits, gates: strengths, s: Math.exp(gf.gain) } : { logits };
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
  function topIdx(a, k) {                                  // indices of the k largest entries, descending
    const idx = Array.from({ length: a.length }, (_, i) => i);
    idx.sort((i, j) => a[j] - a[i]);
    return idx.slice(0, k);
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
      const wantExplain = req.explain !== false;
      const rng = mulberry32(+req.seed || 0);
      model.reset();
      let last = null;
      for (let i = 0; i < ids.length; i++) last = model.step(ids[i], { explain: wantExplain && i === ids.length - 1 });
      const hist = ids.slice(), seen = new Set();
      for (let i = 0; i + 2 < hist.length; i++) seen.add(hist[i] + ',' + hist[i + 1] + ',' + hist[i + 2]);
      const steps = [], pi = wantExplain ? model.profiles() : null;
      for (let n = 0; n < nNew; n++) {
        const logits = last.logits, pAll = softmax(logits, 1);
        const order = topIdx(logits, Math.min(topK, logits.length)), kth = logits[order[order.length - 1]];
        const lg = Array.from(logits, (v) => (v < kth ? -Infinity : v));
        const probs = softmax(lg, temp);
        let u = rng(), tk = order[0], acc = 0;
        for (let i = 0; i < probs.length; i++) { acc += probs[i]; if (u < acc) { tk = i; break; } }
        const step = {
          text: tok.piece(tk), p: rnd(pAll[tk], 4),
          top: topIdx(pAll, 5).map((i) => ({ text: tok.piece(i), p: rnd(pAll[i], 4) })),
        };
        if (wantExplain && last.gates) {
          step.rules = []; step.gates = [];
          last.gates.forEach((g, l) => {
            step.gates.push(Array.from(g, (v) => rnd(v, 2)));
            const contrib = Array.from(g, (gr, r) => gr * last.s * pi[l][r * model.V + tk]);
            for (const r of topIdx(contrib.map(Math.abs), 3)) {
              const sign = g[r] >= 0 ? 1 : -1, prof = Array.from({ length: model.V }, (_, v) => pi[l][r * model.V + v] * sign);
              step.rules.push({ layer: l, rule: r, gate: rnd(g[r], 3), contribution: rnd(contrib[r], 3), raises: topIdx(prof, 5).map((v) => tok.piece(v)) });
            }
          });
        }
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

  const FLM = { Tokenizer, Model, AllFuzzyModel, Api, mulberry32, softmax };
  if (typeof module !== 'undefined' && module.exports) module.exports = FLM; else root.FLM = FLM;
})(typeof window !== 'undefined' ? window : globalThis);
