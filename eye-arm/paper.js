// One physical ink raster is both displayed and sensed. Attention cannot write it.
export const RETINA = 48;
export const PAPER_SIZE = 192;
export const INK_RADIUS = 0.005;
const X = 0.26,
  Y = 0.2,
  SPAN = 0.48;
export const pixelPoint = (i) => [
  X + (((i % RETINA) + 0.5) / RETINA) * SPAN,
  Y + ((Math.floor(i / RETINA) + 0.5) / RETINA) * SPAN,
];
export function pixelAt([x, y]) {
  const col = Math.floor(((x - X) / SPAN) * RETINA),
    row = Math.floor(((y - Y) / SPAN) * RETINA);
  return col >= 0 && col < RETINA && row >= 0 && row < RETINA
    ? row * RETINA + col
    : -1;
}
export class DrawingPaper {
  constructor() {
    this.pixels = new Uint8Array(PAPER_SIZE ** 2);
    this.revision = 0;
  }
  at([x, y]) {
    const col = Math.floor(((x - X) / SPAN) * PAPER_SIZE),
      row = Math.floor(((y - Y) / SPAN) * PAPER_SIZE);
    return col >= 0 && col < PAPER_SIZE && row >= 0 && row < PAPER_SIZE
      ? this.pixels[row * PAPER_SIZE + col]
      : 0;
  }
  stroke(a, b) {
    const scale = PAPER_SIZE / SPAN,
      steps = Math.max(
        1,
        Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1]) * scale * 2),
      ),
      radius = INK_RADIUS * scale;
    for (let t = 0; t <= steps; t++) {
      const x = (a[0] + ((b[0] - a[0]) * t) / steps - X) * scale,
        y = (a[1] + ((b[1] - a[1]) * t) / steps - Y) * scale;
      for (
        let row = Math.max(0, Math.floor(y - radius));
        row <= Math.min(PAPER_SIZE - 1, Math.ceil(y + radius));
        row++
      )
        for (
          let col = Math.max(0, Math.floor(x - radius));
          col <= Math.min(PAPER_SIZE - 1, Math.ceil(x + radius));
          col++
        )
          if (Math.hypot(col + 0.5 - x, row + 0.5 - y) <= radius)
            this.pixels[row * PAPER_SIZE + col] = 1;
    }
    this.revision++;
  }
  erase([x, y], radius) {
    for (let i = 0; i < this.pixels.length; i++) {
      const px = X + (((i % PAPER_SIZE) + 0.5) / PAPER_SIZE) * SPAN,
        py = Y + ((Math.floor(i / PAPER_SIZE) + 0.5) / PAPER_SIZE) * SPAN;
      if (Math.hypot(px - x, py - y) < radius) this.pixels[i] = 0;
    }
    this.revision++;
  }
}
