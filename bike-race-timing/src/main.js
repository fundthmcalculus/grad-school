import { listVideoInputs, startCamera } from './camera.js';
import { WebGpuPreprocessor } from './webgpuPreprocess.js';
import { OcrDetector } from './ocrDetector.js';
import { NumberTracker } from './tracker.js';
import { FootageRecorder, triggerDownload } from './recorder.js';
import { buildSessionJson, buildSessionCsv, triggerTextDownload } from './exporter.js';

const $ = (id) => document.getElementById(id);

const video = $('video');
const overlay = $('overlay');
const cropCanvas = $('cropCanvas');
const gpuCanvas = $('gpuCanvas');
const deviceSelect = $('deviceSelect');
const gpuPill = $('gpuPill');
const sessionPill = $('sessionPill');
const countPill = $('countPill');
const statusLine = $('statusLine');
const logBody = $('logBody');

const btnStartCamera = $('btnStartCamera');
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

const overlayCtx = overlay.getContext('2d');
const cropCtx = cropCanvas.getContext('2d');
const ocrCanvas = document.createElement('canvas');
const ocrCtx = ocrCanvas.getContext('2d');

const gpu = new WebGpuPreprocessor();
let gpuFallbackCtx = null;

let stream = null;
let roiFrac = null;
let dragStart = null;

let detector = null;
let tracker = null;
let recorder = null;
let running = false;
let loopHandle = null;
let lastSession = null;
let lastVideoBlob = null;

function setStatus(msg, level = '') {
  statusLine.textContent = msg;
  statusLine.className = 'status-line' + (level ? ` ${level}` : '');
}

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
    setStatus(`Camera error: ${e.message}`, 'err');
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
});

function applyDefaultRoiIfNeeded() {
  if (roiFrac || !video.videoWidth) return;
  roiFrac = { x: 0.15, y: 0.4, w: 0.7, h: 0.2 };
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
  setStatus('Loading OCR model (first run downloads ~2MB of language data)…');

  detector = new OcrDetector({
    whitelist: whitelistInput.value || '0123456789',
    minConfidence: Number(minConfidenceInput.value) || 0,
  });
  try {
    await detector.init();
  } catch (e) {
    setStatus(`Failed to load OCR engine: ${e.message}`, 'err');
    btnStart.disabled = false;
    return;
  }

  tracker = new NumberTracker({
    missThreshold: Number(missThresholdInput.value) || 3,
    onFinalize: refreshLogTable,
  });

  recorder = new FootageRecorder();
  recorder.start(stream);

  running = true;
  btnStop.disabled = false;
  sessionPill.textContent = 'recording';
  sessionPill.classList.add('on');
  setStatus('Session running.');
  loopStep();
});

btnStop.addEventListener('click', async () => {
  running = false;
  clearTimeout(loopHandle);
  btnStop.disabled = true;
  setStatus('Stopping…');

  const endEpochMs = Date.now();
  const blob = await recorder.stop();
  tracker.flush();
  refreshLogTable();
  await detector.terminate();

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

  btnDownloadVideo.disabled = false;
  btnDownloadJson.disabled = false;
  btnDownloadCsv.disabled = false;
  btnStart.disabled = false;
  sessionPill.textContent = 'stopped';
  sessionPill.classList.remove('on');
  setStatus(`Session stopped. ${tracker.finalized.length} rider(s) logged.`);
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
    tr.className = r.state;
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
