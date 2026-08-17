# CX TT Timing

A browser-based computer-vision timing aid for cyclocross time trials with
staggered group starts (groups of 3, 15 seconds apart). Bib numbers are
mounted on the handlebars and read from the front as riders approach a
camera at the finish/reading line. Everything runs client-side — no server,
no upload of footage anywhere.

It does three things:

1. **Reads bib numbers** off a live camera feed using OCR, restricted to
   digits, on a small region of interest (ROI) you draw over the video.
2. **Logs each sighting**, tracking numbers across frames: if a number is
   read on contiguous cycles, it's treated as one sighting and the logged
   time is the *last* moment it was legible (not the first).
3. **Records the full camera feed** to a downloadable video file, so you can
   go back and manually verify or correct any entry afterward.

## Quick start

Camera access requires a "secure context" (HTTPS, or `localhost`). The
simplest way to run this locally:

```sh
cd bike-race-timing
python3 -m http.server 8000
# open http://localhost:8000/capture.html
```

Any static file server works — this is plain HTML/JS/CSS with no build step.
Open `capture.html`, pick a camera, drag a box over the video to mark the
reading window, then **Start Session**. **Stop Session** finalizes the log
and unlocks the download buttons (footage, JSON log, CSV log).

Open `review.html` afterward, load the downloaded footage + JSON log, and
click **First**/**Last** next to any entry to jump the video to that moment
— useful for confirming or correcting misreads before publishing results.

## Architecture

```
capture.html / review.html      — pages (no framework, ES modules)
src/camera.js                   — getUserMedia device listing + stream setup
src/webgpuPreprocess.js         — WebGPU render pass: grayscale + contrast/
                                   brightness + optional binarize threshold
src/ocrDetector.js              — Tesseract.js wrapper, digit-only whitelist
src/tracker.js                  — contiguous-sighting state machine
src/recorder.js                 — MediaRecorder wrapper, downloadable footage
src/exporter.js                 — JSON/CSV export
src/main.js                     — capture page wiring (ROI select, loop, UI)
src/review.js                   — review page wiring (seek-by-entry, corrections)
```

Each detection cycle (default every 300ms, configurable):

1. The ROI is cropped out of the live video frame onto a small canvas,
   upscaled for OCR clarity.
2. If available, a WebGPU render pipeline (full-screen triangle + fragment
   shader) grayscales and contrast-stretches that crop, and optionally
   hard-thresholds it to pure black/white — this measurably helps OCR
   accuracy on faded or glare-heavy bib numbers. Falls back to using the
   crop as-is on browsers/devices without WebGPU.
3. Tesseract.js (running in its own worker, so the UI doesn't stall) reads
   digit strings out of the processed crop.
4. The tracker matches those digit strings against currently "active"
   numbers: a match extends the sighting's `lastSeen` time; a new number
   starts a new sighting; a number that disappears for `missThreshold`
   consecutive cycles (default 3, to absorb OCR flicker) is finalized and
   logged with its `lastSeen` timestamp.

The full, unprocessed camera stream is recorded the whole time via
`MediaRecorder`, independent of the ROI/detection pipeline, so the AAR
footage always has full context.

## The overtake caveat

This is a heuristic system, not a photo-finish. When two riders are both
briefly legible in the ROI at once — one overtaking another — OCR typically
reports both bib numbers in the same detection cycle, so each gets its own
independent track. Because the rider being overtaken is the one still
occupying the ROI (closer to the line), their track keeps accumulating a
later `lastSeen` for as long as they remain visible, which naturally gives
them the later logged time — i.e. preference goes to the rider being
overtaken, as intended. But this is a side effect of geometry and OCR
timing, not an explicit rule, so a clean, fast pass where both numbers are
only fleetingly readable can still be misordered. Use the Review page to
check any close or contested finishes against the recorded footage.

Other known limitations: numbers that never come cleanly into focus won't
be read at all; the OCR whitelist assumes plain numeric bibs (no letters);
and lighting changes (direct sun, dusk) will need a contrast/threshold
retune per session — the sliders on the Capture page are there for that.

## Roadmap ideas

- Swap Tesseract.js for a small custom-trained digit-detector model run via
  WebGPU (ONNX Runtime Web or TF.js with the WebGPU backend) for real-time,
  bounding-box-aware multi-number detection — would remove most of the
  overtake ambiguity by tracking each number's box position, not just its
  text.
- Import a start list / seed sheet and auto-compute elapsed time per rider
  (start group + 15s offset vs. logged finish time).
- Multi-camera fusion (start + finish) for full lap/segment timing.
