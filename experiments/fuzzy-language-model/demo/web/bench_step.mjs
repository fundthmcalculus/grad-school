import { createRequire } from 'node:module'; import { readFileSync } from 'node:fs'; import { join } from 'node:path';
const require = createRequire(import.meta.url);
const FLM = require(process.env.FLM_JS || './flm.js');
if (process.env.FLM_NOQ8) FLM.int8Kernels = false;
// node demo/web/bench_step.mjs <webdir> [modelId]: milliseconds per step() over 1500 pieces (FLM_NOQ8=1: float32 kernels; FLM_JS=<old flm.js>: compare engines)
const web = process.argv[2], id = process.argv[3] || 'flm-14m-tinystories';
const man = JSON.parse(readFileSync(join(web, `models/${id}.json`), 'utf8'));
const b = readFileSync(join(web, `models/${id}.bin`)); const m = new FLM.AllFuzzyModel(man, b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
m.reset(); for (let i = 0; i < 100; i++) m.step(i % 500);
const t0 = performance.now(); const N = 1500; m.reset(); for (let i = 0; i < N; i++) m.step((i * 7) % 3000);
console.log('ms/step', ((performance.now() - t0) / N).toFixed(3));
