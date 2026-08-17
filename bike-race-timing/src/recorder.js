// MediaRecorder wrapper for saving full-frame race footage for after-action
// review. Records the raw camera stream (not the cropped/preprocessed ROI)
// so reviewers have full context, not just the reading window.

const PREFERRED_MIME_TYPES = [
  'video/webm;codecs=vp9,opus',
  'video/webm;codecs=vp8,opus',
  'video/webm',
  'video/mp4',
];

function pickMimeType() {
  for (const type of PREFERRED_MIME_TYPES) {
    if (MediaRecorder.isTypeSupported(type)) return type;
  }
  return '';
}

export class FootageRecorder {
  /**
   * @param {object} opts
   * @param {(chunk: Blob) => void} [opts.onChunk] called with each raw chunk as it's produced —
   *   use this to stream chunks to disk live (see src/storage.js).
   * @param {boolean} [opts.keepInMemory] when false, chunks are handed to `onChunk` and then
   *   discarded rather than accumulated, so `stop()` resolves with `null`. Use this once a
   *   live disk-writer is in place, so long sessions don't hold the whole video in RAM twice.
   */
  constructor({ onChunk, keepInMemory = true } = {}) {
    this.onChunk = onChunk;
    this.keepInMemory = keepInMemory;
    this.recorder = null;
    this.chunks = [];
    this.mimeType = '';
    this.startEpochMs = null;
    this.blob = null;
  }

  /** @param {MediaStream} stream */
  start(stream) {
    this.chunks = [];
    this.blob = null;
    this.mimeType = pickMimeType();
    this.recorder = new MediaRecorder(stream, this.mimeType ? { mimeType: this.mimeType } : undefined);
    this.recorder.ondataavailable = (e) => {
      if (!e.data || e.data.size === 0) return;
      if (this.keepInMemory) this.chunks.push(e.data);
      if (this.onChunk) this.onChunk(e.data);
    };
    this.startEpochMs = Date.now();
    this.recorder.start(1000); // gather chunks every second so a crash doesn't lose everything
  }

  /** @returns {Promise<Blob|null>} null when `keepInMemory` is false */
  stop() {
    return new Promise((resolve) => {
      if (!this.recorder) {
        resolve(null);
        return;
      }
      this.recorder.onstop = () => {
        this.blob = this.keepInMemory ? new Blob(this.chunks, { type: this.mimeType || 'video/webm' }) : null;
        resolve(this.blob);
      };
      this.recorder.stop();
    });
  }

  get fileExtension() {
    if (this.mimeType.includes('mp4')) return 'mp4';
    return 'webm';
  }

  downloadFilename(prefix = 'race-footage') {
    const stamp = new Date(this.startEpochMs ?? Date.now()).toISOString().replace(/[:.]/g, '-');
    return `${prefix}-${stamp}.${this.fileExtension}`;
  }
}

export function triggerDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}
