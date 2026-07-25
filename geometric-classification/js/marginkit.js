/* Drawing a separating hyperplane and its street, given (w, b).
 *
 * scikit-learn fits the models; this file only turns the fitted coefficients
 * into geometry. Keeping the two apart is deliberate: an SVM solver in the
 * browser would be a hundred lines of quadratic programming that taught the
 * reader nothing the picture does not already say.
 */

/* Where the line w.x + b = c crosses the box [x0,x1] x [y0,y1].
 * Returns the two endpoints, or null if the line misses the box entirely, which
 * happens for the outer kerbs when the street is very wide. */
export function lineInBox(w, b, c, [x0, x1], [y0, y1]) {
  const pts = [];
  const push = (x, y) => {
    if (x >= x0 - 1e-9 && x <= x1 + 1e-9 && y >= y0 - 1e-9 && y <= y1 + 1e-9) pts.push([x, y]);
  };
  if (Math.abs(w[1]) > 1e-12) {
    push(x0, (c - b - w[0] * x0) / w[1]);
    push(x1, (c - b - w[0] * x1) / w[1]);
  }
  if (Math.abs(w[0]) > 1e-12) {
    push((c - b - w[1] * y0) / w[0], y0);
    push((c - b - w[1] * y1) / w[0], y1);
  }
  if (pts.length < 2) return null;
  /* Corner hits land in the list twice; take the two furthest apart. */
  let best = [pts[0], pts[1]];
  let bestD = -1;
  for (let i = 0; i < pts.length; i++) {
    for (let j = i + 1; j < pts.length; j++) {
      const d = (pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2;
      if (d > bestD) {
        bestD = d;
        best = [pts[i], pts[j]];
      }
    }
  }
  return best;
}

/* The filled street between the two kerbs, as a polygon in data coordinates.
 *
 * lineInBox returns whichever pair of box intersections is furthest apart, in
 * no particular order, so simply reversing the second edge is not enough: if
 * the two kerbs come back with opposite orientations the ring closes as a
 * bowtie and the fill turns into two triangles. Project both onto the direction
 * along the boundary and sort, which pins the orientation regardless of what
 * lineInBox happened to hand back. */
export function streetPolygon(w, b, box) {
  const up = lineInBox(w, b, 1, box[0], box[1]);
  const dn = lineInBox(w, b, -1, box[0], box[1]);
  if (!up || !dn) return null;
  const along = [-w[1], w[0]];
  const key = (p) => p[0] * along[0] + p[1] * along[1];
  const [u0, u1] = up[0] && key(up[0]) <= key(up[1]) ? up : [up[1], up[0]];
  const [d0, d1] = dn[0] && key(dn[0]) <= key(dn[1]) ? dn : [dn[1], dn[0]];
  return [u0, u1, d1, d0];
}

/* Signed distance from the boundary, in the plot's own units. */
export const margin = (w, b, p) => (w[0] * p[0] + w[1] * p[1] + b) / Math.hypot(w[0], w[1]);
