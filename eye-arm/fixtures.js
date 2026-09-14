import { RETINA } from "./brain.js";
// Raster fixtures only: neither stroke coordinates nor target paths enter the brain.
export function imageFixture(name) {
  return Array.from({ length: RETINA ** 2 }, (_, i) => {
    const x = ((i % RETINA) + 0.5) / RETINA - 0.5,
      y = (Math.floor(i / RETINA) + 0.5) / RETINA - 0.5;
    if (name === "square")
      return +(Math.abs(Math.max(Math.abs(x), Math.abs(y)) - 0.3) < 0.025);
    if (name === "two_marks")
      return +(
        (Math.abs(x + 0.24) < 0.025 || Math.abs(x - 0.24) < 0.025) &&
        Math.abs(y) < 0.25
      );
    const angle = Math.atan2(y, x),
      radius = Math.hypot(x, y);
    return +(Math.abs(radius - (0.3 + 0.075 * Math.cos(5 * angle))) < 0.025);
  });
}
