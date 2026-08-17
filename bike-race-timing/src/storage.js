// Live, crash-safe persistence via the File System Access API (Chromium
// browsers). Instead of holding the whole session's video in memory and
// only writing it out when "Stop Session" is clicked, footage chunks and
// finalized log entries are appended straight to disk as they happen — a
// browser crash or tab close mid-race loses at most the last chunk, not the
// whole session.
//
// `supported` is false on browsers without the API (Firefox, Safari); the
// caller should fall back to the in-memory Blob + download-at-the-end path
// in that case (see src/recorder.js).

export class LiveFileWriter {
  constructor() {
    this.supported = typeof window !== 'undefined' && 'showDirectoryPicker' in window;
    this.dirHandle = null;
    this.videoWritable = null;
    this.logWritable = null;
    this.videoFilename = null;
    this.logFilename = null;
    this._videoQueue = Promise.resolve();
    this._logQueue = Promise.resolve();
  }

  /** Must be called directly from a user gesture (e.g. a button click). */
  async pickDirectory() {
    if (!this.supported) return false;
    try {
      this.dirHandle = await window.showDirectoryPicker({ id: 'cx-tt-timing', mode: 'readwrite' });
      return true;
    } catch (e) {
      // User cancelled the picker — not an error.
      return false;
    }
  }

  get directoryName() {
    return this.dirHandle ? this.dirHandle.name : null;
  }

  async openSessionFiles(baseName) {
    if (!this.dirHandle) return false;
    if ((await this.dirHandle.queryPermission({ mode: 'readwrite' })) !== 'granted') {
      const perm = await this.dirHandle.requestPermission({ mode: 'readwrite' });
      if (perm !== 'granted') return false;
    }

    this.videoFilename = `${baseName}.webm`;
    this.logFilename = `${baseName}.jsonl`;

    const videoHandle = await this.dirHandle.getFileHandle(this.videoFilename, { create: true });
    this.videoWritable = await videoHandle.createWritable();

    const logHandle = await this.dirHandle.getFileHandle(this.logFilename, { create: true });
    this.logWritable = await logHandle.createWritable();

    this._videoQueue = Promise.resolve();
    this._logQueue = Promise.resolve();
    return true;
  }

  appendVideoChunk(blob) {
    if (!this.videoWritable) return;
    this._videoQueue = this._videoQueue.then(() => this.videoWritable.write(blob));
  }

  appendLogEntry(obj) {
    if (!this.logWritable) return;
    const line = `${JSON.stringify(obj)}\n`;
    this._logQueue = this._logQueue.then(() => this.logWritable.write(line));
  }

  async close() {
    await this._videoQueue;
    await this._logQueue;
    if (this.videoWritable) {
      await this.videoWritable.close();
      this.videoWritable = null;
    }
    if (this.logWritable) {
      await this.logWritable.close();
      this.logWritable = null;
    }
  }
}
