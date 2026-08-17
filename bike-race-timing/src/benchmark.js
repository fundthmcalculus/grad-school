// On-site performance preflight: runs a burst of real detection cycles
// against the live camera + current ROI/preprocessing settings, and a short
// recording, to measure whether *this* laptop/camera can keep up with the
// configured detection interval — on the operator's actual race-day
// hardware, not a developer's machine. Run this once on-site before racing
// starts (see RACE_DAY.md).

import { OcrDetector } from './ocrDetector.js';
import { looksLikeBlankFrame } from './webgpuPreprocess.js';

function average(arr) {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function percentile(sortedArr, p) {
  const idx = Math.min(sortedArr.length - 1, Math.floor(p * sortedArr.length));
  return sortedArr[idx];
}

/**
 * @param {object} ctx
 * @param {HTMLVideoElement} ctx.video
 * @param {{x:number,y:number,w:number,h:number}} ctx.roiFrac
 * @param {import('./webgpuPreprocess.js').WebGpuPreprocessor} ctx.gpu
 * @param {boolean} [ctx.webgpuEnabled] mirrors the operator's opt-in toggle — WebGPU defaults to off
 * @param {() => object} ctx.gpuOpts
 * @param {HTMLCanvasElement} ctx.cropCanvas
 * @param {CanvasRenderingContext2D} ctx.cropCtx
 * @param {HTMLCanvasElement} ctx.gpuSourceCanvas disposable copy fed to WebGPU — never cropCanvas itself
 * @param {CanvasRenderingContext2D} ctx.gpuSourceCtx
 * @param {HTMLCanvasElement} ctx.gpuCanvas
 * @param {CanvasRenderingContext2D|null} ctx.gpuFallbackCtx
 * @param {HTMLCanvasElement} ctx.ocrCanvas
 * @param {CanvasRenderingContext2D} ctx.ocrCtx
 * @param {string} ctx.whitelist
 * @param {number} ctx.minConfidence
 * @param {object} [opts]
 * @param {number} [opts.cycles]
 * @param {(done: number, total: number) => void} [opts.onProgress]
 */
export async function runDetectionBenchmark(ctx, opts = {}) {
  const { cycles = 15, onProgress } = opts;
  const {
    video, roiFrac, gpu, webgpuEnabled = false, gpuOpts, cropCanvas, cropCtx, gpuSourceCanvas, gpuSourceCtx, gpuCanvas, gpuFallbackCtx, ocrCanvas, ocrCtx,
    whitelist, minConfidence,
  } = ctx;

  const detector = new OcrDetector({ whitelist, minConfidence });
  await detector.init();

  const gpuAvailableAtStart = webgpuEnabled && gpu.available;
  let gpuFailedAnyCycle = false;

  const preprocessTimes = [];
  const ocrTimes = [];
  const cycleTimes = [];

  try {
    for (let i = 0; i < cycles; i += 1) {
      const t0 = performance.now();

      const vw = video.videoWidth;
      const vh = video.videoHeight;
      cropCtx.drawImage(video, roiFrac.x * vw, roiFrac.y * vh, roiFrac.w * vw, roiFrac.h * vh, 0, 0, cropCanvas.width, cropCanvas.height);

      const t1 = performance.now();
      const attemptedGpu = webgpuEnabled && gpu.available;
      if (attemptedGpu) gpuSourceCtx.drawImage(cropCanvas, 0, 0);
      let gpuOk = attemptedGpu && gpu.process(gpuSourceCanvas, gpuOpts());
      if (gpuOk) {
        ocrCtx.drawImage(gpuCanvas, 0, 0);
        if (looksLikeBlankFrame(ocrCtx)) {
          gpu.reportInvalidFrame('blank frame after a reported-successful render');
          gpuOk = false;
        }
      }
      if (attemptedGpu && !gpuOk) gpuFailedAnyCycle = true;
      if (!gpuOk) {
        if (gpuFallbackCtx) gpuFallbackCtx.drawImage(cropCanvas, 0, 0);
        ocrCtx.drawImage(cropCanvas, 0, 0);
      }
      const t2 = performance.now();

      await detector.detect(ocrCanvas);
      const t3 = performance.now();

      preprocessTimes.push(t2 - t0);
      ocrTimes.push(t3 - t2);
      cycleTimes.push(t3 - t0);
      if (onProgress) onProgress(i + 1, cycles);
    }
  } finally {
    await detector.terminate();
  }

  const gpuDisabledDuringRun = gpuAvailableAtStart && !gpu.available;

  const sortedCycles = [...cycleTimes].sort((a, b) => a - b);
  const p50 = percentile(sortedCycles, 0.5);
  const p95 = percentile(sortedCycles, 0.95);
  const max = sortedCycles[sortedCycles.length - 1];

  return {
    cycles,
    canvasWidth: cropCanvas.width,
    canvasHeight: cropCanvas.height,
    preprocessMsAvg: average(preprocessTimes),
    ocrMsAvg: average(ocrTimes),
    cycleMsP50: p50,
    cycleMsP95: p95,
    cycleMsMax: max,
    // Recommend the interval be at least p95 with 30% headroom, rounded up
    // to the nearest 10ms, so a slow cycle doesn't eat into the next one.
    recommendedIntervalMs: Math.ceil((p95 * 1.3) / 10) * 10,
    gpuAvailableAtStart,
    gpuFailedAnyCycle,
    gpuDisabledDuringRun,
  };
}

/**
 * Records a short clip from the live stream to estimate real-world encoded
 * bitrate (and therefore disk usage per hour) on this camera/lighting, since
 * this varies a lot with scene content and can't be predicted generically.
 * @param {MediaStream} stream
 * @param {number} [durationMs]
 */
export function measureBitrate(stream, durationMs = 4000) {
  return new Promise((resolve, reject) => {
    const mimeCandidates = ['video/webm;codecs=vp9,opus', 'video/webm;codecs=vp8,opus', 'video/webm'];
    const mimeType = mimeCandidates.find((t) => MediaRecorder.isTypeSupported(t)) || '';
    let rec;
    try {
      rec = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    } catch (e) {
      reject(e);
      return;
    }
    let bytes = 0;
    rec.ondataavailable = (e) => {
      if (e.data) bytes += e.data.size;
    };
    rec.onstop = () => {
      const bytesPerSec = bytes / (durationMs / 1000);
      resolve({
        bytesPerSec,
        estGbPerHour: (bytesPerSec * 3600) / 1024 ** 3,
      });
    };
    rec.onerror = (e) => reject(e.error || new Error('MediaRecorder error'));
    rec.start(500);
    setTimeout(() => rec.stop(), durationMs);
  });
}
