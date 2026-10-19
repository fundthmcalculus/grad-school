// node demo/web/test_parity.mjs : browser engine vs the torch model (logits) and vs the HF tokenizer (ids)
import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
const require = createRequire(import.meta.url);
const FLM = require('./flm.js');
const here = dirname(fileURLToPath(import.meta.url));
const J = (p) => JSON.parse(readFileSync(join(here, p), 'utf8'));
const buf = (p) => { const b = readFileSync(join(here, p)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
let bad = 0;
const tok = new FLM.Tokenizer(J('tokenizer.json'));
for (const c of J('tokenizer.check.json')) {
  const got = tok.encode(c.text), ok = JSON.stringify(got) === JSON.stringify(c.ids);
  const dec = got.map((i) => tok.piece(i)).join('');
  const dok = dec === c.decoded;
  if (!ok || !dok) { bad++; console.log('TOKENIZER MISMATCH', JSON.stringify(c.text), ok ? '' : `ids ${got} vs ${c.ids}`, dok ? '' : `decode ${JSON.stringify(dec)} vs ${JSON.stringify(c.decoded)}`); }
}
console.log('tokenizer: ' + J('tokenizer.check.json').length + ' strings checked');
for (const m of J('models/index.json')) {
  const man = J(`models/${m.id}.json`), chk = J(`models/${m.id}.check.json`);
  const model = new FLM.Model(man, buf(`models/${m.id}.bin`));
  let last; const t0 = performance.now();
  for (const id of chk.ids) last = model.step(id, { explain: true });
  const ms = performance.now() - t0;
  let md = 0, mx = 0; chk.last_logits.forEach((v, i) => { md = Math.max(md, Math.abs(v - last.logits[i])); mx = Math.max(mx, Math.abs(v)); });
  const tol = man.dtype === 'float16' ? 0.15 : 2e-3;
  const arg = (a) => a.indexOf(Math.max(...a));
  const same = arg(chk.last_logits) === arg(Array.from(last.logits));
  const ok = md < tol && same;
  if (!ok) bad++;
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${m.id}: max |dlogit| ${md.toExponential(2)} (tol ${tol}, |logit| up to ${mx.toFixed(1)}), argmax ${same ? 'same' : 'DIFFERENT'}, ${(ms / chk.ids.length).toFixed(2)} ms/token`);
}
process.exit(bad ? 1 : 0);
