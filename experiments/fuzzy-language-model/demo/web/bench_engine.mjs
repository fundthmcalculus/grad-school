// node demo/web/bench_engine.mjs [webdir] [modelId] : time the browser engine's step() in node (milliseconds per piece; the profile build of the rule view)
import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
const require = createRequire(import.meta.url);
const FLM = require('./flm.js');
const here = dirname(fileURLToPath(import.meta.url));
const web = resolve(process.argv[2] || here), id = process.argv[3] || 'flm-allfuzzy-f7';
const J = (p) => JSON.parse(readFileSync(join(web, p), 'utf8'));
const buf = (p) => { const b = readFileSync(join(web, p)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const man = J(`models/${id}.json`);
let t0 = performance.now();
const model = new FLM.AllFuzzyModel(man, buf(`models/${man.id}.bin`));
console.log(`load ${(performance.now() - t0).toFixed(0)} ms (${man.dtype || 'float32'})`);
const ids = Array.from({ length: 80 }, (_, i) => (i * 37 + 11) % man.config.vocab_size);
for (const explain of [false, true]) {
  model.reset(); for (const t of ids.slice(0, 4)) model.step(t, { explain });   // warm up
  model.reset(); t0 = performance.now();
  for (const t of ids) model.step(t, { explain });
  console.log(`step explain=${explain}: ${((performance.now() - t0) / ids.length).toFixed(2)} ms per piece`);
}
t0 = performance.now(); model.pi = null; if (model.profiles) model.profiles(); console.log(`profiles(): ${(performance.now() - t0).toFixed(0)} ms`);
// the whole Api.generate (tokenizer, sampling, DRY, rule view), 60 pieces
const api = new FLM.Api(web, async (p) => JSON.parse(readFileSync(p, 'utf8')), async (p) => { const b = readFileSync(p); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); });
for (const explain of [false, true]) {
  t0 = performance.now();
  const r = await api.generate({ model: id, prompt: 'Once upon a time', max_tokens: 60, temperature: 0.5, top_k: 10, seed: 0, explain, dry: true });
  console.log(`api.generate explain=${explain}: ${(performance.now() - t0).toFixed(0)} ms for ${r.steps.length} pieces`);
}
