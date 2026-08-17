# Vendored third-party code

This directory vendors pre-built assets from the Tesseract.js project so the
capture app works with **no internet access** at the race venue (Tesseract.js
otherwise fetches these from a CDN at runtime).

| File(s) | From | Version |
|---|---|---|
| `tesseract.min.js` | [`tesseract.js`](https://github.com/naptha/tesseract.js) | 5.1.0 |
| `worker.min.js` | `tesseract.js` | 5.1.0 |
| `tesseract-core-lstm.wasm.js`, `tesseract-core-simd-lstm.wasm.js` | [`tesseract.js-core`](https://github.com/naptha/tesseract.js-core) | matching `tesseract.js` 5.1.0 |
| `lang/eng.traineddata.gz` | [`@tesseract.js-data/eng`](https://www.npmjs.com/package/@tesseract.js-data) (the `4.0.0_best_int` / LSTM variant) | 1.0.0 |

All of the above are distributed under the Apache License 2.0. See
`tesseract.js.LICENSE.md` and `tesseract.js-core.LICENSE` in this directory
for the full license text. The trained-data file is the standard Tesseract
OCR English model, also Apache 2.0 licensed by the Tesseract OCR project.

To add another language, download the matching `<lang>.traineddata.gz`
("best"/LSTM variant, e.g. from `@tesseract.js-data/<lang>` on npm or
jsdelivr) into `lang/`, then pass `{ lang: '<lang>' }` to `OcrDetector` in
`src/main.js`.
