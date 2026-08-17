// Contiguous-sighting tracker for bib numbers.
//
// Each detection cycle reports the set of numbers currently readable in the
// ROI. A number is "active" from the first cycle it appears in until it goes
// missing for `missThreshold` consecutive cycles (debounces OCR flicker). The
// logged time for a rider is `lastSeen` — the last moment the number was
// legible — per the race-timing rule that we want when the rider stopped
// being visible, not when we first noticed them.
//
// Overtakes: if rider A (in front) is passed by rider B while both numbers
// are briefly legible, OCR usually reports both numbers independently in the
// same cycle, so each gets its own track. Because A is still occupying the
// ROI (closer to the line) after B's number becomes readable, A's track
// naturally keeps accumulating a later `lastSeen`, which is the "give
// preference to the rider being overtaken" behavior described in the spec.
// This is a heuristic, not a guarantee — it can still be fooled by a clean
// pass where both numbers are only briefly legible.

export class NumberTracker {
  constructor({ missThreshold = 3, onFinalize } = {}) {
    this.missThreshold = missThreshold;
    this.onFinalize = onFinalize;
    this.active = new Map();
    this.finalized = [];
  }

  /**
   * @param {Array<{number: string, confidence: number}>} detections
   * @param {number} timestamp epoch ms
   */
  observe(detections, timestamp) {
    const seenNow = new Set();

    for (const d of detections) {
      if (!d.number) continue;
      seenNow.add(d.number);

      const track = this.active.get(d.number);
      if (track) {
        track.lastSeen = timestamp;
        track.hits += 1;
        track.misses = 0;
        track.bestConfidence = Math.max(track.bestConfidence, d.confidence ?? 0);
      } else {
        this.active.set(d.number, {
          number: d.number,
          firstSeen: timestamp,
          lastSeen: timestamp,
          hits: 1,
          misses: 0,
          bestConfidence: d.confidence ?? 0,
        });
      }
    }

    for (const [number, track] of this.active) {
      if (seenNow.has(number)) continue;
      track.misses += 1;
      if (track.misses >= this.missThreshold) {
        this.active.delete(number);
        this._finalize(track);
      }
    }
  }

  /** Finalize everything still active (call when a session ends). */
  flush() {
    for (const track of this.active.values()) {
      this._finalize(track);
    }
    this.active.clear();
  }

  /** Snapshot of currently-active (not yet finalized) tracks, for live UI. */
  activeSnapshot() {
    return Array.from(this.active.values()).map((t) => ({ ...t }));
  }

  _finalize(track) {
    const record = {
      number: track.number,
      firstSeen: track.firstSeen,
      lastSeen: track.lastSeen,
      durationMs: track.lastSeen - track.firstSeen,
      hits: track.hits,
      confidence: track.bestConfidence,
    };
    this.finalized.push(record);
    if (this.onFinalize) this.onFinalize(record);
  }
}
