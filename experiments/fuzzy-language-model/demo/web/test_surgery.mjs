// node demo/web/test_surgery.mjs : the browser's rule surgery (surgery.js) against the numpy reference edit (flm/surgery_check.py -> surgery.check.json)
import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
const require = createRequire(import.meta.url);
const FLM = require('./flm.js');
const Surgery = require('./surgery.js');
const here = dirname(fileURLToPath(import.meta.url));
const J = (p) => JSON.parse(readFileSync(join(here, p), 'utf8'));
const buf = (p) => { const b = readFileSync(join(here, p)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const man = J('models/flm-allfuzzy-f7.json');
const model = new FLM.AllFuzzyModel(man, buf('models/flm-allfuzzy-f7.bin'));
const argmax = (a) => { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; };
let bad = 0;
for (const [n, c] of J('surgery.check.json').entries()) {
  model.reset();
  let r; for (const id of c.prefix) r = model.step(id, { explain: true });
  const last = model.L - 1, before = Float32Array.from(r.logits);
  const t0 = performance.now();
  const plan = Surgery.plan(model, r.xPre, r.gates[last], c.target, { k: r.gates[last].length, margin: 0.5 });
  const ms = performance.now() - t0;
  const relOk = Math.abs(plan.relNorm - c.rel_norm) / c.rel_norm < 0.02;   // the same minimum-norm edit as numpy (the line search and gradient step agree to ~1%)
  const reach = plan.ok && plan.after >= 0.5 - 1e-6;
  // apply: a fresh replay of the same prefix must now put the wanted piece first; undo must restore the scores exactly
  Surgery.apply(model, plan);
  model.reset(); let r2; for (const id of c.prefix) r2 = model.step(id);
  const top1 = argmax(r2.logits) === c.target;
  Surgery.undo(model, plan);
  model.reset(); let r3; for (const id of c.prefix) r3 = model.step(id);
  let dmax = 0; for (let i = 0; i < before.length; i++) dmax = Math.max(dmax, Math.abs(r3.logits[i] - before[i]));
  const ok = relOk && reach && top1 && dmax < 1e-4;
  if (!ok) bad++;
  console.log(`${ok ? 'ok  ' : 'FAIL'} edit ${n}: relNorm js ${plan.relNorm.toFixed(4)} np ${c.rel_norm.toFixed(4)}, margin ${plan.before.toFixed(2)} -> ${plan.after.toFixed(2)}, top1 after apply ${top1}, undo restores scores to ${dmax.toExponential(1)}, plan ${ms.toFixed(0)} ms`);
}
process.exit(bad ? 1 : 0);
