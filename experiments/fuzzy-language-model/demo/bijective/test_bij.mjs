// node demo/bijective/test_bij.mjs <models dir> <id> : the browser engine's scores against torch's (compared as log-probabilities, so the per-position constant of the Gaussian drops out)
import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
const require = createRequire(import.meta.url);
const BIJ = require('./bij.js');
const dir = resolve(process.argv[2] || 'demo/bijective/models'), id = process.argv[3] || 'bij-10k-c';
const man = JSON.parse(readFileSync(join(dir, id + '.json'), 'utf8')), chk = JSON.parse(readFileSync(join(dir, id + '.check.json'), 'utf8'));
const b = readFileSync(join(dir, id + '.bin'));
const m = new BIJ.BijModel(man, b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
m.reset(); let lg; for (const t of chk.ids) lg = m.step(t).logits;
const ls = (x) => { const mx = Math.max(...x); const z = Math.log(x.reduce((s, v) => s + Math.exp(v - mx), 0)); return Array.from(x, (v) => v - mx - z); };
const a = ls(lg), r = ls(chk.last_logits);
let err = 0, am = 0, ar = 0; for (let i = 0; i < a.length; i++) { err = Math.max(err, Math.abs(a[i] - r[i])); if (a[i] > a[am]) am = i; if (r[i] > r[ar]) ar = i; }
console.log(`${id} (arm ${man.bij.arm}): max |dlogprob| ${err.toExponential(2)}, argmax ${am === ar ? 'same' : 'DIFFERENT'} (${am}/${ar})`);
const t0 = performance.now(); BIJ.greedy(m, BIJ.Chars.encode('Once upon a time'), 200); console.log(`200 characters in ${(performance.now() - t0).toFixed(0)} ms`);
process.exit(err < 2e-3 && am === ar ? 0 : 1);
