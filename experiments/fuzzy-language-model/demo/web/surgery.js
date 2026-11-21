/* Rule surgery for the in-browser all-fuzzy model (PLAN_SURGERY.md; the numpy reference is flm/surgery_exp.py).
 *
 * At a position where the model chose the wrong next piece, name the piece that should have come next; the surgeon finds the smallest change to
 * the consequent vectors of the strongest-firing feed-forward rules of the LAST block that makes it the top choice by a margin, applies it to the
 * model's weights in memory, and can undo it.  Why that block: the output scores are an exact function of the residual x before the final gain
 * controller, x = x_mid + sum_r s_r c_r (s_r the rule's strength, c_r its consequent), and nothing after that block reads the bank's output, so an
 * edit changes a position's scores only through the strengths s_r of the rules it touches, and cannot change what later positions compute.
 *
 *   const plan = FLM.Surgery.plan(model, r.xPre, r.gates[model.L - 1], targetId, { k: 3, margin: 0.5 });   // r = model.step(id, {explain: true})
 *   if (plan.ok) FLM.Surgery.apply(model, plan);     // edits the last block's consequents in memory; regenerate from the position
 *   FLM.Surgery.undo(model, plan);                    // restores them exactly
 */
(function (root, factory) {
  const S = factory();
  if (typeof module === 'object' && module.exports) module.exports = S;
  if (root && root.FLM) root.FLM.Surgery = S; else if (root) root.FLM_Surgery = S;
})(typeof window !== 'undefined' ? window : (typeof globalThis !== 'undefined' ? globalThis : this), function () {
  function scores(model, x) {                       // output scores for a residual x: the final gain controller, then the tied table
    const { d, V, E } = model, hn = model.gain(x, 'norm.').out, lg = new Float32Array(V);
    for (let v = 0; v < V; v++) { let a = 0; const r = v * d; for (let i = 0; i < d; i++) a += E[r + i] * hn[i]; lg[v] = a; }
    return lg;
  }
  function marginAt(model, x, t) {                  // logit of t minus the best other logit
    const lg = scores(model, x); let best = -Infinity, arg = -1;
    for (let v = 0; v < lg.length; v++) if (v !== t && lg[v] > best) { best = lg[v]; arg = v; }
    return { margin: lg[t] - best, rival: arg, lg };
  }
  function direction(model, x, t, p) {              // d(logit_t - logit_p)/dx, central differences (the step is relative to |x_j|, as in the numpy reference)
    const { d, E } = model, g = new Float64Array(d), de = new Float64Array(d);
    for (let i = 0; i < d; i++) de[i] = E[t * d + i] - E[p * d + i];
    const xp = Float64Array.from(x), xm = Float64Array.from(x);
    for (let j = 0; j < d; j++) {
      const e = 1e-3 * Math.max(1, Math.abs(x[j]));
      xp[j] = x[j] + e; xm[j] = x[j] - e;
      const a = model.gain(xp, 'norm.').out, b = model.gain(xm, 'norm.').out;
      let s = 0; for (let i = 0; i < d; i++) s += de[i] * (a[i] - b[i]);
      g[j] = s / (2 * e); xp[j] = x[j]; xm[j] = x[j];
    }
    return g;
  }
  const norm = (v) => Math.sqrt(v.reduce((a, b) => a + b * b, 0));

  /** plan an edit: { ok, rules:[{layer, rule, strength, dC: Float64Array}], dx, before, after, relNorm } */
  function plan(model, xPre, strengths, target, opts) {
    const k = (opts && opts.k) || 3, margin = (opts && opts.margin) || 0.5, d = model.d, layer = model.L - 1;
    const x = Float64Array.from(xPre), before = marginAt(model, x, target);
    if (before.margin >= margin) return { ok: true, noop: true, rules: [], dx: new Float64Array(d), before: before.margin, after: before.margin, relNorm: 0 };
    const g = direction(model, x, target, before.rival), gn = norm(g) || 1, u = g.map((v) => v / gn), xn = norm(x);
    const at = (a) => marginAt(model, x.map((v, i) => v + a * u[i]), target).margin;
    let lo = 0, hi = 0.05 * xn;
    while (at(hi) < margin) { lo = hi; hi *= 2; if (hi > 50 * xn) return { ok: false, reason: 'no step up to 50x |x| reaches the margin', before: before.margin }; }
    for (let i = 0; i < 30; i++) { const mid = 0.5 * (lo + hi); if (at(mid) < margin) lo = mid; else hi = mid; }
    const dx = u.map((v) => v * hi), s = Float64Array.from(strengths), R = s.length;
    const order = Array.from({ length: R }, (_, r) => r).sort((a, b) => Math.abs(s[b]) - Math.abs(s[a])).slice(0, Math.min(k, R));
    let ss = 0; for (const r of order) ss += s[r] * s[r];
    const cq = model.W['blocks.' + layer + '.ffn.conseq'].data, rules = []; let dn = 0, cn = 0;
    for (const r of order) {
      const dC = new Float64Array(d); let rn = 0;
      for (let i = 0; i < d; i++) { dC[i] = (s[r] / ss) * dx[i]; dn += dC[i] * dC[i]; const c = cq[r * d + i]; cn += c * c; }
      rules.push({ layer, rule: r, strength: s[r], dC });
    }
    const xe = x.map((v, i) => { let a = v; for (const e of rules) a += e.strength * e.dC[i]; return a; });
    const after = marginAt(model, xe, target);
    return { ok: after.margin >= margin - 1e-6, rules, dx, before: before.margin, after: after.margin, relNorm: Math.sqrt(dn / cn), target };
  }
  function shift(model, plan_, sign) {
    for (const e of plan_.rules) {
      const cq = model.W['blocks.' + e.layer + '.ffn.conseq'].data, d = model.d;
      for (let i = 0; i < d; i++) cq[e.rule * d + i] += sign * e.dC[i];
      if (model.requantRow) model.requantRow(e.layer, e.rule);   // int8 kernels: rebuild this rule's row
    }
    model.pi = null; model.rows = null;            // the rule profiles are derived from the consequents
  }
  const apply = (model, p) => shift(model, p, +1);
  const undo = (model, p) => shift(model, p, -1);
  const toJSON = (p) => ({ target: p.target, relNorm: p.relNorm, rules: p.rules.map((e) => ({ layer: e.layer, rule: e.rule, dC: Array.from(e.dC) })) });
  const fromJSON = (j) => ({ target: j.target, relNorm: j.relNorm, rules: j.rules.map((e) => ({ layer: e.layer, rule: e.rule, strength: 0, dC: Float64Array.from(e.dC) })) });
  return { plan, apply, undo, toJSON, fromJSON, scores, marginAt };
});
