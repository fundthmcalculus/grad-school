// WebGPU preprocessing pass: grayscale + contrast/brightness + optional
// hard threshold (binarize), rendered straight into a <canvas> via a
// full-screen-triangle render pipeline. This is the "step towards WebGPU"
// piece of the pipeline — it cleans up the cropped ROI before handing it to
// the OCR engine, which noticeably helps recognition on low-contrast bib
// numbers (glare, faded print, motion blur edges).
//
// Falls back cleanly (available === false) on browsers without WebGPU; the
// caller should use the plain 2D-canvas crop directly in that case.

const SHADER_SRC = `
struct Uniforms {
  contrast: f32,
  brightness: f32,
  threshold: f32,
  binarize: f32,
};
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> u: Uniforms;

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VSOut {
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0)
  );
  var out: VSOut;
  out.pos = vec4<f32>(pos[idx], 0.0, 1.0);
  out.uv = vec2<f32>((pos[idx].x + 1.0) * 0.5, 1.0 - (pos[idx].y + 1.0) * 0.5);
  return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let c = textureSample(tex, samp, in.uv);
  let gray = dot(c.rgb, vec3<f32>(0.299, 0.587, 0.114));
  var v = (gray - 0.5) * u.contrast + 0.5 + u.brightness;
  v = clamp(v, 0.0, 1.0);
  if (u.binarize > 0.5) {
    v = select(0.0, 1.0, v > u.threshold);
  }
  return vec4<f32>(v, v, v, 1.0);
}
`;

// Failures are treated as recoverable-until-proven-otherwise: a bad frame,
// a driver hiccup, or (as observed in at least one headless/software-WebGPU
// environment during testing) a canvas-context render pass that leaves the
// device unable to do further copyExternalImageToTexture calls. Rather than
// erroring out every single detection cycle for the rest of a race, this
// class disables itself after a few consecutive failures so the caller can
// fall back to the plain 2D-canvas crop — which has no such failure mode —
// and keeps OCR running instead of silently going dark.
const MAX_CONSECUTIVE_FAILURES = 2;

/**
 * Cheap post-hoc sanity check for a canvas the caller just drew a
 * `WebGpuPreprocessor` output into: the fragment shader above always writes
 * fully-opaque pixels, so alpha === 0 anywhere means nothing was actually
 * rendered (the "silent blank frame" failure mode `process()`'s return
 * value alone can't catch). Checking one pixel is enough and effectively
 * free — cheaper than a single OCR call by several orders of magnitude.
 * @param {CanvasRenderingContext2D} ctx a 2D context already drawn into via drawImage(gpuCanvas, ...)
 */
export function looksLikeBlankFrame(ctx) {
  const { data } = ctx.getImageData(0, 0, 1, 1);
  return data[3] === 0;
}

export class WebGpuPreprocessor {
  constructor() {
    this.available = false;
    this.device = null;
    this.context = null;
    this.pipeline = null;
    this.sampler = null;
    this.uniformBuffer = null;
    this.canvas = null;
    this.format = 'rgba8unorm';
    this.consecutiveFailures = 0;
    this.lastError = null;
    /** Called once, the moment self-disable happens after repeated failures. */
    this.onDisabled = null;
  }

  /** @param {HTMLCanvasElement} canvas an offscreen canvas dedicated to WebGPU output */
  async init(canvas) {
    if (!navigator.gpu) return false;

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return false;

    this.device = await adapter.requestDevice();
    this.device.lost.then((info) => {
      if (!this.available) return; // already disabled deliberately
      this._disable(`GPU device lost (${info.reason}): ${info.message}`);
    });
    this.canvas = canvas;
    this.context = canvas.getContext('webgpu');
    if (!this.context) return false;

    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({
      device: this.device,
      format: this.format,
      alphaMode: 'opaque',
    });

    const shaderModule = this.device.createShaderModule({ code: SHADER_SRC });

    this.uniformBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.sampler = this.device.createSampler({ magFilter: 'linear', minFilter: 'linear' });

    this.pipeline = this.device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: shaderModule, entryPoint: 'vs_main' },
      fragment: { module: shaderModule, entryPoint: 'fs_main', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list' },
    });

    this.available = true;
    return true;
  }

  /**
   * Render a processed copy of `source` into the WebGPU canvas. Returns
   * false (without throwing) on failure — the caller should treat that
   * exactly like `available === false` for this cycle, i.e. use the 2D
   * crop directly instead.
   * @param {CanvasImageSource} source
   * @param {{contrast?: number, brightness?: number, threshold?: number, binarize?: boolean}} opts
   */
  process(source, opts = {}) {
    if (!this.available) return false;
    const { contrast = 1.4, brightness = 0.0, threshold = 0.5, binarize = false } = opts;
    const { device, context, canvas } = this;
    const width = canvas.width;
    const height = canvas.height;

    let texture;
    try {
      texture = device.createTexture({
        size: [width, height, 1],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      });

      device.queue.copyExternalImageToTexture({ source }, { texture }, [width, height]);

      const uniformData = new Float32Array([contrast, brightness, threshold, binarize ? 1 : 0]);
      device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

      const bindGroup = device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.sampler },
          { binding: 1, resource: texture.createView() },
          { binding: 2, resource: { buffer: this.uniformBuffer } },
        ],
      });

      const encoder = device.createCommandEncoder();
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: context.getCurrentTexture().createView(),
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
      });
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.draw(3);
      pass.end();
      device.queue.submit([encoder.finish()]);
    } catch (e) {
      this._recordFailure(e.message);
      return false;
    } finally {
      if (texture) texture.destroy();
    }

    this.consecutiveFailures = 0;
    return true;
  }

  /**
   * Some failure modes here don't throw: in at least one headless/software-
   * WebGPU environment seen during testing, the device silently stopped
   * actually rendering (leaving the canvas blank/transparent) shortly before
   * reporting itself lost, even though the render-pass calls themselves
   * didn't throw. `process()` returning `true` therefore isn't a complete
   * correctness guarantee — the caller should sanity-check the resulting
   * pixels (e.g. alpha === 255, since the shader always writes opaque
   * output) and call this if the frame looks blank, so it counts the same
   * as a thrown error toward the self-disable threshold.
   * @param {string} reason
   */
  reportInvalidFrame(reason) {
    this._recordFailure(reason);
  }

  _recordFailure(reason) {
    this.consecutiveFailures += 1;
    this.lastError = reason;
    if (this.consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
      this._disable(`repeated WebGPU preprocessing failures (${reason})`);
    }
  }

  _disable(reason) {
    this.available = false;
    this.lastError = reason;
    if (this.onDisabled) this.onDisabled(reason);
  }
}
