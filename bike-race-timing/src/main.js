import { listVideoInputs, startCamera } from './camera.js';
import { WebGpuPreprocessor } from './webgpuPreprocess.js';
import { OcrDetector } from './ocrDetector.js';
import { NumberTracker } from './tracker.js';
import { FootageRecorder, triggerDownload } from './recorder.js';
import { buildSessionJson, buildSessionCsv, triggerTextDownload } from './exporter.js';
import { LiveFileWriter } from './storage.js';
import { loadSettings, saveSettings } from './settings.js';

const $ = (id) => document.getElementById(id);

const video = $('video');
const overlay = $('overlay');
const cropCanvas = $('cropCanvas');
const gpuCanvas = $('gpuCanvas');
const deviceSelect = $('deviceSelect');
const gpuPill = $('gpuPill');
const sessionPill = $('sessionPill');
const elapsedPill = $('elapsedPill');
const folderPill = $('folderPill');
const countPill = $('countPill');
const statusLine = $('statusLine');
const logBody = $('logBody');

const btnStartCamera = $('btnStartCamera');
const btnChooseFolder = $('btnChooseFolder');
const btnStart = $('btnStart');
const btnStop = $('btnStop');
const btnDownloadVideo = $('btnDownloadVideo');
const btnDownloadJson = $('btnDownloadJson');
const btnDownloadCsv = $('btnDownloadCsv');

const intervalMsInput = $('intervalMs');
const missThresholdInput = $('missThreshold');
const minConfidenceInput = $('minConfidence');
const contrastInput = $('contrast');
const brightnessInput = $('brightness');
const binarizeInput = $('binarize');
const thresholdInput = $('threshold');
const whitelistInput = $('whitelist');
const upscaleInput = $('upscale');
const beepEnabledInput = $('beepEnabled');

const overlayCtx = overlay.getContext('2d');
const cropCtx = cropCanvas.getContext('2d');
const ocrCanvas = document.createElement('canvas');
const ocrCtx = ocrCanvas.getContext('2d');

const gpu = new WebGpuPreprocessor();
let gpuFallbackCtx = null;

const fileWriter = new LiveFileWriter();
let useLiveFiles = false;

let stream = null;
let roiFrac = null;
let dragStart = null;

let detector = null;
let tracker = null;
let recorder = null;
let running = false;
let loopHandle = null;
let elapsedHandle = null;
let lastSession = null;
let lastVideoBlob = null;
let audioCtx = null;
const recentlyFinalized = new Set();

function setStatus(msg, level = '') {
  statusLine.textContent = msg;
  statusLine.className = 'status-line' + (level ? ` ${level}` : '');
}

window.addEventListener('unhandledrejection', (e) => {
  const msg = e.reason && e.reason.message ? e.reason.message : String(e.reason);
  setStatus(`Unexpected error: ${msg}`, 'err');
});

// ---- Settings persistence -------------------------------------------------

function collectSettings() {
  return {
    intervalMs: Number(intervalMsInput.value),
    missThreshold: Number(missThresholdInput.value),
    minConfidence: Number(minConfidenceInput.value),
    contrast: Number(contrastInput.value),
    brightness: Number(brightnessInput.value),
    binarize: binarizeInput.checked,
    threshold: Number(thresholdInput.value),
    whitelist: whitelistInput.value,
    upscale: Number(upscaleInput.value),
    beepEnabled: beepEnabledInput.checked,
    roi: roiFrac,
  };
}

function persistSettings() {
  saveSettings(collectSettings());
}

function applySavedSettings(saved) {
  if (!saved) return;
  if (saved.intervalMs != null) intervalMsInput.value = saved.intervalMs;
  if (saved.missThreshold != null) missThresholdInput.value = saved.missThreshold;
  if (saved.minConfidence != null) minConfidenceInput.value = saved.minConfidence;
  if (saved.contrast != null) contrastInput.value = saved.contrast;
  if (saved.brightness != null) brightnessInput.value = saved.brightness;
  if (saved.binarize != null) binarizeInput.checked = saved.binarize;
  if (saved.threshold != null) thresholdInput.value = saved.threshold;
  if (saved.whitelist != null) whitelistInput.value = saved.whitelist;
  if (saved.upscale != null) upscaleInput.value = saved.upscale;
  if (saved.beepEnabled != null) beepEnabledInput.checked = saved.beepEnabled;
  if (saved.roi) roiFrac = saved.roi;
}

const savedSettings = loadSettings();
applySavedSettings(savedSettings);

[intervalMsInput, missThresholdInput, minConfidenceInput, contrastInput, brightnessInput,
  binarizeInput, thresholdInput, whitelistInput, beepEnabledInput].forEach((el) => {
  el.addEventListener('change', persistSettings);
});
upscaleInput.addEventListener('change', () => {
  persistSettings();
  updateCropCanvasSizeFromRoi();
});

// ---- Camera setup ----------------------------------------------------

async function populateDevices() {
  try {
    const devices = await listVideoInputs();
    const prevValue = deviceSelect.value;
    deviceSelect.innerHTML = '';
    devices.forEach((d, i) => {
      const opt = document.createElement('option');
      opt.value = d.deviceId;
      opt.textContent = d.label || `Camera ${i + 1}`;
      deviceSelect.appendChild(opt);
    });
    if (prevValue) deviceSelect.value = prevValue;
  } catch (e) {
    // enumerateDevices can fail before permission is granted on some browsers; ignore.
  }
}

btnStartCamera.addEventListener('click', async () => {
  try {
    stream = await startCamera(video, deviceSelect.value || null);
    await populateDevices();
    setStatus(`Camera started: ${video.videoWidth}x${video.videoHeight}`);
    fitOverlayToVideo();
    applyDefaultRoiIfNeeded();
  } catch (e) {
    setStatus(`Camera error: ${e.message}. Check camera permissions and that no other app is using it.`, 'err');
  }
});

video.addEventListener('loadedmetadata', () => {
  fitOverlayToVideo();
  applyDefaultRoiIfNeeded();
});
window.addEventListener('resize', fitOverlayToVideo);

// ---- ROI selection -----------------------------------------------------

function fitOverlayToVideo() {
  const rect = video.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) return;
  overlay.width = rect.width;
  overlay.height = rect.height;
  drawRoiOverlay();
}

function rectFromPoints(a, b) {
  const x = Math.min(a.x, b.x);
  const y = Math.min(a.y, b.y);
  const w = Math.abs(b.x - a.x);
  const h = Math.abs(b.y - a.y);
  return { x, y, w, h };
}

function drawRoiOverlay(previewRectPx = null) {
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  const rectPx = previewRectPx || (roiFrac && {
    x: roiFrac.x * overlay.width,
    y: roiFrac.y * overlay.height,
    w: roiFrac.w * overlay.width,
    h: roiFrac.h * overlay.height,
  });
  if (!rectPx) return;
  overlayCtx.strokeStyle = '#4fd1c5';
  overlayCtx.lineWidth = 2;
  overlayCtx.strokeRect(rectPx.x, rectPx.y, rectPx.w, rectPx.h);
  overlayCtx.fillStyle = 'rgba(79, 209, 197, 0.12)';
  overlayCtx.fillRect(rectPx.x, rectPx.y, rectPx.w, rectPx.h);
}

overlay.addEventListener('mousedown', (e) => {
  const r = overlay.getBoundingClientRect();
  dragStart = { x: e.clientX - r.left, y: e.clientY - r.top };
});
overlay.addEventListener('mousemove', (e) => {
  if (!dragStart) return;
  const r = overlay.getBoundingClientRect();
  const cur = { x: e.clientX - r.left, y: e.clientY - r.top };
  drawRoiOverlay(rectFromPoints(dragStart, cur));
});
window.addEventListener('mouseup', (e) => {
  if (!dragStart) return;
  const r = overlay.getBoundingClientRect();
  const cur = { x: e.clientX - r.left, y: e.clientY - r.top };
  const rectPx = rectFromPoints(dragStart, cur);
  dragStart = null;
  if (rectPx.w < 12 || rectPx.h < 12) {
    drawRoiOverlay();
    return;
  }
  roiFrac = {
    x: rectPx.x / overlay.width,
    y: rectPx.y / overlay.height,
    w: rectPx.w / overlay.width,
    h: rectPx.h / overlay.height,
  };
  drawRoiOverlay();
  updateCropCanvasSizeFromRoi();
  persistSettings();
});

function applyDefaultRoiIfNeeded() {
  if (!video.videoWidth) return;
  if (!roiFrac) roiFrac = { x: 0.15, y: 0.4, w: 0.7, h: 0.2 };
  drawRoiOverlay();
  updateCropCanvasSizeFromRoi();
}

function updateCropCanvasSizeFromRoi() {
  if (!roiFrac || !video.videoWidth) return;
  const upscale = Number(upscaleInput.value) || 2;
  const roiPxW = roiFrac.w * video.videoWidth;
  const roiPxH = roiFrac.h * video.videoHeight;
  const w = Math.min(1024, Math.max(64, Math.round(roiPxW * upscale)));
  const h = Math.min(384, Math.max(24, Math.round(roiPxH * upscale)));
  cropCanvas.width = w;
  cropCanvas.height = h;
  gpuCanvas.width = w;
  gpuCanvas.height = h;
  ocrCanvas.width = w;
  ocrCanvas.height = h;
}

// ---- WebGPU init --------------------------------------------------------

(async () => {
  const ok = await gpu.init(gpuCanvas);
  if (ok) {
    gpuPill.textContent = 'WebGPU: enabled';
    gpuPill.classList.add('on');
  } else {
    gpuPill.textContent = 'WebGPU: unavailable (CPU fallback)';
    gpuFallbackCtx = gpuCanvas.getContext('2d');
  }
})();

// ---- Save folder (crash-safe live persistence) ---------------------------

if (!fileWriter.supported) {
  btnChooseFolder.disabled = true;
  folderPill.textContent = 'live-save unsupported in this browser (footage downloads at end)';
} else {
  btnChooseFolder.addEventListener('click', async () => {
    const ok = await fileWriter.pickDirectory();
    if (ok) {
      useLiveFiles = true;
      folderPill.textContent = `folder: ${fileWriter.directoryName} (footage + log stream live)`;
      folderPill.classList.add('on');
      setStatus('Save folder selected. Footage and log entries will be written live during the session.');
    }
  });
}

// ---- Detection loop -------------------------------------------------------

function captureRoiToCropCanvas() {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const sx = roiFrac.x * vw;
  const sy = roiFrac.y * vh;
  const sw = roiFrac.w * vw;
  const sh = roiFrac.h * vh;
  cropCtx.drawImage(video, sx, sy, sw, sh, 0, 0, cropCanvas.width, cropCanvas.height);
}

function gpuOptsFromUi() {
  return {
    contrast: Number(contrastInput.value),
    brightness: Number(brightnessInput.value),
    threshold: Number(thresholdInput.value),
    binarize: binarizeInput.checked,
  };
}

async function loopStep() {
  if (!running) return;
  try {
    if (roiFrac && video.videoWidth) {
      captureRoiToCropCanvas();
      if (gpu.available) {
        gpu.process(cropCanvas, gpuOptsFromUi());
        ocrCtx.drawImage(gpuCanvas, 0, 0);
      } else {
        if (gpuFallbackCtx) gpuFallbackCtx.drawImage(cropCanvas, 0, 0);
        ocrCtx.drawImage(cropCanvas, 0, 0);
      }
      const detections = await detector.detect(ocrCanvas);
      tracker.observe(detections, Date.now());
      refreshLogTable();
    }
  } catch (err) {
    setStatus(`Detection error: ${err.message}`, 'err');
  }
  if (running) {
    loopHandle = setTimeout(loopStep, Number(intervalMsInput.value) || 300);
  }
}

// ---- Audio cue ------------------------------------------------------------

function beep() {
  if (!beepEnabledInput.checked || !audioCtx) return;
  try {
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.frequency.value = 880;
    gain.gain.setValueAtTime(0.15, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + 0.15);
    osc.connect(gain).connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.16);
  } catch (e) {
    // Autoplay-policy or unsupported-browser edge cases — non-critical, ignore.
  }
}

// ---- Elapsed timer ----------------------------------------------------

function formatElapsed(ms) {
  const totalSeconds = Math.floor(ms / 1000);
  const m = Math.floor(totalSeconds / 60).toString().padStart(2, '0');
  const s = (totalSeconds % 60).toString().padStart(2, '0');
  return `${m}:${s}`;
}

// ---- Session lifecycle ----------------------------------------------------

btnStart.addEventListener('click', async () => {
  if (!stream) {
    setStatus('Start the camera first.', 'warn');
    return;
  }
  if (!roiFrac) {
    setStatus('Draw a reading window (ROI) on the video first.', 'warn');
    return;
  }
  btnStart.disabled = true;

  try {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === 'suspended') await audioCtx.resume();
  } catch (e) {
    audioCtx = null;
  }

  const baseName = `race-session-${new Date().toISOString().replace(/[:.]/g, '-')}`;
  if (useLiveFiles) {
    setStatus('Opening session files on disk…');
    const opened = await fileWriter.openSessionFiles(baseName);
    if (!opened) {
      setStatus('Could not open files in the chosen folder (permission lost?). Falling back to in-memory recording for this session.', 'warn');
      useLiveFiles = false;
    }
  }

  setStatus('Loading OCR model…');
  detector = new OcrDetector({
    whitelist: whitelistInput.value || '0123456789',
    minConfidence: Number(minConfidenceInput.value) || 0,
  });
  try {
    await detector.init();
  } catch (e) {
    setStatus(`Failed to load OCR engine: ${e.message}. Check that vendor/tesseract/ is present next to this page.`, 'err');
    btnStart.disabled = false;
    return;
  }

  recentlyFinalized.clear();
  tracker = new NumberTracker({
    missThreshold: Number(missThresholdInput.value) || 3,
    onFinalize: (record) => {
      refreshLogTable();
      beep();
      recentlyFinalized.add(record.number);
      setTimeout(() => {
        recentlyFinalized.delete(record.number);
        refreshLogTable();
      }, 1500);
      if (useLiveFiles) {
        fileWriter.appendLogEntry({
          number: record.number,
          firstSeenEpochMs: record.firstSeen,
          lastSeenEpochMs: record.lastSeen,
          durationMs: record.durationMs,
          hits: record.hits,
          confidence: record.confidence,
        });
      }
    },
  });

  recorder = new FootageRecorder({
    keepInMemory: !useLiveFiles,
    onChunk: useLiveFiles ? (chunk) => fileWriter.appendVideoChunk(chunk) : undefined,
  });
  recorder.start(stream);

  running = true;
  btnStop.disabled = false;
  sessionPill.textContent = 'recording';
  sessionPill.classList.add('on');
  elapsedPill.style.display = '';
  elapsedHandle = setInterval(() => {
    elapsedPill.textContent = formatElapsed(Date.now() - recorder.startEpochMs);
  }, 500);
  setStatus(useLiveFiles
    ? `Session running. Footage streaming to ${fileWriter.videoFilename}.`
    : 'Session running.');
  loopStep();
});

btnStop.addEventListener('click', async () => {
  running = false;
  clearTimeout(loopHandle);
  clearInterval(elapsedHandle);
  btnStop.disabled = true;
  setStatus('Stopping…');

  const endEpochMs = Date.now();
  const blob = await recorder.stop();
  tracker.flush();
  refreshLogTable();
  await detector.terminate();
  if (useLiveFiles) await fileWriter.close();

  lastSession = {
    recordingStartEpochMs: recorder.startEpochMs,
    recordingEndEpochMs: endEpochMs,
    roi: roiFrac,
    settings: {
      intervalMs: Number(intervalMsInput.value),
      missThreshold: Number(missThresholdInput.value),
      minConfidence: Number(minConfidenceInput.value),
      whitelist: whitelistInput.value,
    },
    entries: tracker.finalized.slice(),
  };
  lastVideoBlob = blob;
  window.__lastSession = lastSession; // convenience for console debugging

  btnDownloadVideo.disabled = !lastVideoBlob;
  btnDownloadJson.disabled = false;
  btnDownloadCsv.disabled = false;
  btnStart.disabled = false;
  sessionPill.textContent = 'stopped';
  sessionPill.classList.remove('on');
  elapsedPill.style.display = 'none';

  if (useLiveFiles) {
    setStatus(`Session stopped. ${tracker.finalized.length} rider(s) logged. Footage saved to "${fileWriter.videoFilename}" and log to "${fileWriter.logFilename}" in your chosen folder.`);
  } else {
    setStatus(`Session stopped. ${tracker.finalized.length} rider(s) logged.`);
  }
});

btnDownloadVideo.addEventListener('click', () => {
  if (!lastVideoBlob) return;
  triggerDownload(lastVideoBlob, recorder.downloadFilename());
});
btnDownloadJson.addEventListener('click', () => {
  if (!lastSession) return;
  triggerTextDownload(buildSessionJson(lastSession), 'race-log.json', 'application/json');
});
btnDownloadCsv.addEventListener('click', () => {
  if (!lastSession) return;
  triggerTextDownload(buildSessionCsv(lastSession), 'race-log.csv', 'text/csv');
});

window.addEventListener('beforeunload', (e) => {
  if (running) {
    e.preventDefault();
    e.returnValue = '';
  }
});

// ---- Live log table --------------------------------------------------------

function fmtTime(ms) {
  const d = new Date(ms);
  return d.toLocaleTimeString(undefined, { hour12: false }) + '.' + String(d.getMilliseconds()).padStart(3, '0');
}

function refreshLogTable() {
  const rows = [];
  if (tracker) {
    for (const t of tracker.activeSnapshot()) {
      rows.push({ ...t, state: 'active' });
    }
    for (const t of tracker.finalized) {
      rows.push({ ...t, state: 'finalized' });
    }
  }
  rows.sort((a, b) => b.lastSeen - a.lastSeen);

  logBody.innerHTML = '';
  for (const r of rows) {
    const tr = document.createElement('tr');
    tr.className = r.state + (recentlyFinalized.has(r.number) ? ' flash' : '');
    tr.innerHTML = `
      <td>${r.number}</td>
      <td>${fmtTime(r.firstSeen)}</td>
      <td>${fmtTime(r.lastSeen)}</td>
      <td>${r.hits}</td>
      <td>${Math.round(r.bestConfidence ?? r.confidence ?? 0)}</td>
      <td>${r.state}</td>
    `;
    logBody.appendChild(tr);
  }
  countPill.textContent = `${rows.length} rider(s)`;
}

// ---- Init --------------------------------------------------------------

populateDevices();
