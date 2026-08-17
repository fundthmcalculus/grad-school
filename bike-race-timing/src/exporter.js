// Export helpers for a finished timing session.

/**
 * @param {object} session
 * @param {number} session.recordingStartEpochMs
 * @param {{x:number,y:number,w:number,h:number}} session.roi fractional ROI used
 * @param {Array} session.entries finalized tracker records
 */
export function buildSessionJson(session) {
  return JSON.stringify(
    {
      version: 1,
      recordingStartEpochMs: session.recordingStartEpochMs,
      recordingEndEpochMs: session.recordingEndEpochMs ?? null,
      roi: session.roi ?? null,
      settings: session.settings ?? {},
      entries: session.entries.map((e) => ({
        number: e.number,
        firstSeenEpochMs: e.firstSeen,
        lastSeenEpochMs: e.lastSeen,
        durationMs: e.durationMs,
        hits: e.hits,
        confidence: e.confidence,
        confirmed: e.confirmed ?? undefined,
      })),
    },
    null,
    2,
  );
}

export function buildSessionCsv(session) {
  const header = 'number,first_seen_iso,last_seen_iso,last_seen_video_offset_s,duration_ms,hits,confidence,confirmed\n';
  const rows = session.entries
    .slice()
    .sort((a, b) => a.lastSeen - b.lastSeen)
    .map((e) => {
      const offsetS = ((e.lastSeen - session.recordingStartEpochMs) / 1000).toFixed(2);
      return [
        e.number,
        new Date(e.firstSeen).toISOString(),
        new Date(e.lastSeen).toISOString(),
        offsetS,
        e.durationMs,
        e.hits,
        e.confidence.toFixed(1),
        e.confirmed === undefined ? '' : e.confirmed,
      ].join(',');
    });
  return header + rows.join('\n') + '\n';
}

export function triggerTextDownload(text, filename, mimeType = 'application/json') {
  const blob = new Blob([text], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}
