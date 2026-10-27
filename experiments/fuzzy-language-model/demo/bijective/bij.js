/* Browser inference for the invertible-meaning-layer model (flm/run_bijective.py; PLAN_BIJECTIVE.md).
 *
 * The blocks are the all-fuzzy blocks that web/flm.js already runs (AllFuzzyModel); this file replaces only the ends of the model:
 *   embedding   z_v = F(C_v) is a table computed at export time (it arrives in the one-tier slot of the engine), so the lookup is the engine's own;
 *   output      the likelihood of each symbol is a Gaussian around its codeword:  logit_v = -|u - t_v|^2 / (2 tau^2),
 *                 arm c:     u = F^-1(h), t_v = C_v     (the logic space routed back through the same fuzzy layer)
 *                 arm ci:    u = F^-1(G^-1(h)), t_v = C_v   (as c, with the exactly invertible gain G undone first; the embedding table already holds G(F(C)))
 *                 arm z:     u = h,       t_v = z_v
 *                 arm ident: u = h,       t_v = C_v     (control: no invertible layer)
 *                 arm tied:  the usual dot product with the embedding table (the engine's own scores)
 *   F^-1 is exact: each coupling changed half the coordinates, b' = b exp(s(a)) + t(a), with s and t order-1 TSK systems of the other half a, so the inverse is
 *   b = (b' - t(a)) exp(-s(a)) with the same two fuzzy systems evaluated on a.
 * Greedy decoding only (the highest-likelihood symbol); no sampling.
 */
(function (root) {
  'use strict';
  const FLM = typeof module !== 'undefined' && module.exports ? require('../web/flm.js') : root.FLM;

  // symbols: 0 end of story, 1 unknown, 2 newline, 3..97 printable ASCII 32..126 (flm/data.py)
  const Chars = {
    V: 98,
    encode(text) { const out = []; for (const ch of text) { const c = ch.codePointAt(0); out.push(c === 10 ? 2 : c >= 32 && c <= 126 ? 3 + c - 32 : 1); } return out; },
    decode(ids) { return ids.map((i) => (i === 0 ? '\n\n' : i === 1 ? '�' : i === 2 ? '\n' : String.fromCharCode(i - 3 + 32))).join(''); },
  };

  function tsk1(W, p, x) {                                 // order-1 TSK system: sum_r w_r (A_r x + c_r), Gaussian rules over x (product t-norm, exponent 1/d_in)
    const c = W[p + 'centers'], lw = W[p + 'log_width'].data, A = W[p + 'A'], cc = W[p + 'c'].data;
    const R = c.shape[0], din = c.shape[1], dout = A.shape[1], cd = c.data, Ad = A.data;
    const lg = new Float64Array(R); let mx = -Infinity;
    for (let r = 0; r < R; r++) { let q = 0; for (let j = 0; j < din; j++) { const df = x[j] - cd[r * din + j]; q += Math.exp(-2 * lw[r * din + j]) * df * df; } lg[r] = -0.5 * q / din; if (lg[r] > mx) mx = lg[r]; }
    let z = 0; for (let r = 0; r < R; r++) { lg[r] = Math.exp(lg[r] - mx); z += lg[r]; }
    const y = new Float64Array(dout);
    for (let r = 0; r < R; r++) {
      const w = lg[r] / z, base = r * dout * din;
      for (let o = 0; o < dout; o++) { let v = cc[r * dout + o]; for (let j = 0; j < din; j++) v += Ad[base + o * din + j] * x[j]; y[o] += w * v; }
    }
    return y;
  }

  class BijModel extends FLM.AllFuzzyModel {
    constructor(manifest, buf) {
      super(manifest, buf);
      const b = (this.bij = manifest.bij), W = this.W, d = this.d, V = this.V;
      this.arm = b.arm;
      if (b.arm === 'tied') return;
      this.C = b.arm === 'z' ? null : W['codebook'].data;             // codewords
      if (b.arm === 'ci') { this.logw = W['G.log_w'].data; this.vg = W['G.vg'].data; this.gG = b.g_grid; this.gLo = b.g_lo; this.gDu = (b.g_hi - b.g_lo) / (b.g_grid - 1); }
      this.Z = this.embH;                                              // effective embedding rows (V x d)
      this.itau2 = Math.exp(-2 * W['log_tau'].data[0]);                // 1 / tau^2
      this.nC = b.couplings;
      this.smax = b.s_max;
      this.target = b.arm === 'z' ? this.Z : W['codebook'].data;
      this.tn = new Float64Array(V); for (let v = 0; v < V; v++) { let s = 0; for (let i = 0; i < d; i++) s += this.target[v * d + i] ** 2; this.tn[v] = s; }
    }
    inverse(y) {                                                       // F^-1, the couplings in reverse order (flip on odd couplings)
      const d = y.length;
      for (let k = this.nC - 1; k >= 0; k--) {
        const flip = k % 2 === 1, na = flip ? d - (d >> 1) : d >> 1, nb = d - na;
        const a = flip ? y.slice(nb) : y.slice(0, na), yb = flip ? y.slice(0, nb) : y.slice(na);
        const sh = tsk1(this.W, `F.steps.${k}.shift_sys.`, a), sc = tsk1(this.W, `F.steps.${k}.scale_sys.`, a);
        const b = new Float64Array(nb);
        for (let i = 0; i < nb; i++) b[i] = (yb[i] - sh[i]) * Math.exp(-this.smax * Math.tanh(sc[i] / this.smax));
        y = new Float64Array(d);
        if (flip) { y.set(b, 0); y.set(a, nb); } else { y.set(a, 0); y.set(b, na); }
      }
      return y;
    }
    gainInverse(y) {                                                   // G^-1: y' = y / w, v = log rms y', u = v^-1(v) on the same grid segments the forward used, x = y' exp(u - v)
      const d = y.length, yp = new Float64Array(d); let ss = 0;
      for (let i = 0; i < d; i++) { yp[i] = y[i] / Math.exp(this.logw[i]); ss += yp[i] * yp[i]; }
      const v = 0.5 * Math.log(Math.max(ss / d, 1e-60)), vg = this.vg, G = this.gG;
      let lo = 0, hi = G; while (lo < hi) { const m = (lo + hi) >> 1; if (vg[m] >= v) hi = m; else lo = m + 1; }   // first index with vg >= v (torch.searchsorted, left)
      const i = Math.min(Math.max(lo - 1, 0), G - 2), u = this.gLo + i * this.gDu + (v - vg[i]) / (vg[i + 1] - vg[i]) * this.gDu, f = Math.exp(u - v);
      for (let k = 0; k < d; k++) yp[k] *= f;
      return yp;
    }
    step(tok, opts) {
      const r = super.step(tok, { explain: true });                    // the engine's own blocks; r.xPre is the state before the final norm
      if (this.arm === 'tied') return opts && opts.explain ? r : { logits: r.logits };
      const d = this.d, V = this.V, go = this.arm === 'ci' ? this.gainInverse(r.xPre) : this.gain(r.xPre, 'norm.').out;
      const u = this.arm === 'c' || this.arm === 'ci' ? this.inverse(go) : go, T = this.target, logits = new Float32Array(V);
      for (let v = 0; v < V; v++) { let dot = 0; const o = v * d; for (let i = 0; i < d; i++) dot += u[i] * T[o + i]; logits[v] = (dot - 0.5 * this.tn[v]) * this.itau2; }   // -|u - t|^2 / (2 tau^2) up to a constant of the position
      return opts && opts.explain ? { ...r, logits } : { logits };
    }
  }

  function greedy(model, promptIds, n, onStep) {                       // the highest-likelihood symbol each step; returns the new ids and each one's probability
    model.reset();
    let logits = model.step(0).logits;                                 // every story starts after an end-of-story symbol
    for (const t of promptIds) logits = model.step(t).logits;
    const ids = [], probs = [];
    for (let k = 0; k < n && model.pos < model.maxLen; k++) {
      let best = 0, mx = -Infinity; for (let i = 0; i < logits.length; i++) if (logits[i] > mx) { mx = logits[i]; best = i; }
      let z = 0; for (let i = 0; i < logits.length; i++) z += Math.exp(logits[i] - mx);
      ids.push(best); probs.push(1 / z);
      if (onStep) onStep(best, 1 / z);
      logits = model.step(best).logits;
    }
    return { ids, probs };
  }

  const API = { BijModel, Chars, greedy };
  if (typeof module !== 'undefined' && module.exports) module.exports = API; else root.BIJ = API;
})(typeof window !== 'undefined' ? window : globalThis);
