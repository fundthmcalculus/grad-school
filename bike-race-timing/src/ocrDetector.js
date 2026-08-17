// Thin wrapper around Tesseract.js (loaded globally via a vendored <script>
// in capture.html — see vendor/tesseract/) configured to read digit-only bib
// numbers from a small cropped canvas. Runs in a worker internally, so it
// doesn't block the UI thread; the caller is responsible for not overlapping
// calls (see `busy` below).
//
// All engine/language assets are loaded from vendor/tesseract/ rather than
// Tesseract.js's default jsdelivr CDN — race venues frequently have no
// internet access, so nothing here may depend on it. If you need a
// different language, download its <lang>.traineddata.gz (lstm/"best_int"
// variant) into vendor/tesseract/lang/ and pass it as `lang`.

/* global Tesseract */

const VENDOR_BASE = new URL('../vendor/tesseract/', import.meta.url).href;

export class OcrDetector {
  constructor({ whitelist = '0123456789', minConfidence = 40, lang = 'eng', logger } = {}) {
    this.whitelist = whitelist;
    this.minConfidence = minConfidence;
    this.lang = lang;
    this.logger = logger || (() => {});
    this.worker = null;
    this.busy = false;
  }

  async init() {
    this.worker = await Tesseract.createWorker(this.lang, 1, {
      workerPath: `${VENDOR_BASE}worker.min.js`,
      corePath: VENDOR_BASE,
      langPath: `${VENDOR_BASE}lang/`,
      gzip: true,
      logger: this.logger,
    });
    await this.worker.setParameters({
      tessedit_char_whitelist: this.whitelist,
      tessedit_pageseg_mode: '11', // sparse text: find as many text fragments as possible
    });
  }

  /**
   * @param {HTMLCanvasElement} canvas
   * @returns {Promise<Array<{number: string, confidence: number, bbox: object}>>}
   */
  async detect(canvas) {
    if (!this.worker || this.busy) return [];
    this.busy = true;
    try {
      const { data } = await this.worker.recognize(canvas);
      const words = data.words || [];
      return words
        .map((w) => ({
          number: (w.text || '').replace(/\D/g, ''),
          confidence: w.confidence,
          bbox: w.bbox,
        }))
        .filter((w) => w.number.length > 0 && w.confidence >= this.minConfidence);
    } finally {
      this.busy = false;
    }
  }

  async terminate() {
    if (this.worker) {
      await this.worker.terminate();
      this.worker = null;
    }
  }
}
