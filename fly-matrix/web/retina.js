// Rendered RGB pixels drive the retained photoreceptor neurons. No object labels,
// fruit positions, body headings or target bearings enter this encoder.
// We supply a fixed index-grid and luminance rule: the payload has an aggregate
// photoreceptor mask but no optical axes or R1–R8/left/right receptor mapping.
// This candidate is not recovered anatomical retinotopy or calibrated fly vision.
export const RETINA_WIDTH = 64, RETINA_HEIGHT = 32;
export const RETINA_MAX_WIDTH = 256, RETINA_MAX_HEIGHT = 128;
export const RETINA_MAPPING = "supplied-retained-index-grid/v1";
export const RETINA_LUMINANCE = Object.freeze([0.2126, 0.7152, 0.0722]);

export function validateRetinalFrame(frame) {
  if (!frame || typeof frame !== "object" || Array.isArray(frame)) throw new TypeError("invalid_retinal_frame");
  if (Object.hasOwn(frame, "indices") || Object.hasOwn(frame, "levels")) throw new TypeError("retinal_recipients_are_worker_owned");
  const { width, height, rgba, origin } = frame;
  if (!Number.isSafeInteger(width) || width < 1 || width > RETINA_MAX_WIDTH
    || !Number.isSafeInteger(height) || height < 1 || height > RETINA_MAX_HEIGHT)
    throw new RangeError("invalid_retinal_dimensions");
  if (!(rgba instanceof Uint8Array || rgba instanceof Uint8ClampedArray) || rgba.length !== width * height * 4)
    throw new TypeError("invalid_retinal_rgba");
  if (origin !== "bottom-left" && origin !== "top-left") throw new TypeError("invalid_retinal_origin");
  return frame;
}

/** Compile the worker-owned photoreceptor mask into a deterministic pixel map.
 * Sorted retained-neuron indices fill balanced rows, left-to-right/top-to-bottom.
 * Neighboring grid sites need not be neighboring biological receptive fields.
 * No atlas soma positions are interpreted as optical viewing directions. */
export function createRetina({ n, photoreceptors } = {}) {
  if (!Number.isSafeInteger(n) || n < 1 || n > 0x7fffffff) throw new RangeError("invalid_retinal_neuron_count");
  if ((!Array.isArray(photoreceptors) && !ArrayBuffer.isView(photoreceptors))
    || !Number.isSafeInteger(photoreceptors.length) || photoreceptors.length < 1 || photoreceptors.length > n)
    throw new TypeError("missing_retinal_photoreceptors");
  const members = Array.from(photoreceptors);
  if (!members.every(i => Number.isSafeInteger(i) && i >= 0 && i < n)
    || new Set(members).size !== members.length) throw new RangeError("invalid_retinal_photoreceptors");
  members.sort((a, b) => a - b);
  const indices = Int32Array.from(members), count = indices.length;
  const rows = Math.max(1, Math.round(Math.sqrt(count / (RETINA_WIDTH / RETINA_HEIGHT))));
  const perRow = Math.floor(count / rows), extra = count % rows;
  const uv = new Float64Array(2 * count);
  for (let row = 0, k = 0; row < rows; row++) {
    const columns = perRow + (row < extra ? 1 : 0);
    for (let col = 0; col < columns; col++, k++) {
      uv[2 * k] = (col + 0.5) / columns;
      uv[2 * k + 1] = (row + 0.5) / rows;
    }
  }
  const mapping = Object.freeze({ scheme: RETINA_MAPPING, count, rows,
    minColumns: perRow, maxColumns: perRow + (extra ? 1 : 0),
    aspect: RETINA_WIDTH / RETINA_HEIGHT,
    transduction: "bilinear rendered RGB luminance; no adaptation or fitted spectral sensitivity" });

  function encode(frame) {
    const { width, height, rgba, origin } = validateRetinalFrame(frame);
    const levels = new Float32Array(count);
    const luminance = (x, y) => {
      const offset = 4 * (y * width + x);
      return (RETINA_LUMINANCE[0] * rgba[offset] + RETINA_LUMINANCE[1] * rgba[offset + 1]
        + RETINA_LUMINANCE[2] * rgba[offset + 2]) / 255;
    };
    for (let k = 0; k < count; k++) {
      const x = uv[2 * k] * (width - 1);
      const v = uv[2 * k + 1], y = (origin === "bottom-left" ? 1 - v : v) * (height - 1);
      const x0 = Math.floor(x), y0 = Math.floor(y), x1 = Math.min(width - 1, x0 + 1), y1 = Math.min(height - 1, y0 + 1);
      const dx = x - x0, dy = y - y0;
      levels[k] = (1 - dy) * ((1 - dx) * luminance(x0, y0) + dx * luminance(x1, y0))
        + dy * ((1 - dx) * luminance(x0, y1) + dx * luminance(x1, y1));
    }
    // Copies may be transferred across threads without detaching this map.
    return { indices: indices.slice(), levels };
  }
  return Object.freeze({ mapping, encode });
}
