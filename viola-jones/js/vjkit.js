/* Viola-Jones in the browser: the integral image, Haar features evaluated
 * from it, and the trained cascade run over a real photograph.
 *
 * Every rectangle sum is four array lookups, which is the entire point of the
 * method, so the code is written to make that visible rather than hidden
 * behind a library call.
 */

/* Summed-area table with a zero row and column: ii[r][c] is the sum of every
 * pixel strictly above and to the left of (r, c). Same convention as the
 * generator's numpy version. */
export function integralImage(gray, w, h) {
  const ii = new Float64Array((w + 1) * (h + 1));
  for (let r = 0; r < h; r++) {
    let rowSum = 0;
    for (let c = 0; c < w; c++) {
      rowSum += gray[r * w + c];
      ii[(r + 1) * (w + 1) + (c + 1)] = ii[r * (w + 1) + (c + 1)] + rowSum;
    }
  }
  return ii;
}

/* The four lookups. Rectangle corners are inclusive, in window coordinates,
 * offset by (r0, c0) so one integral image serves every sliding position. */
export function rectSum(ii, stride, top, left, r0, c0, r1, c1) {
  const a = (top + r0) * stride + (left + c0);
  const b = (top + r0) * stride + (left + c1 + 1);
  const c = (top + r1 + 1) * stride + (left + c0);
  const d = (top + r1 + 1) * stride + (left + c1 + 1);
  return ii[d] - ii[b] - ii[c] + ii[a];
}

/* A Haar feature: rectangles with alternating signs, the FIRST one negative,
 * which is scikit-image's convention and the one the generator verified. */
export function featureValue(ii, stride, top, left, rects) {
  let total = 0;
  for (let k = 0; k < rects.length; k++) {
    const [r0, c0, r1, c1] = rects[k];
    const s = rectSum(ii, stride, top, left, r0, c0, r1, c1);
    total += k % 2 === 0 ? -s : s;
  }
  return total;
}

export const stumpVote = (value, learner) =>
  ((learner.polarity === 1 ? value > learner.threshold : value <= learner.threshold) ? 1 : -1);

/* Run the cascade at one window position. Returns the stage it died at (or
 * the number of stages if it survived) and how many features were evaluated,
 * which is the quantity the whole method exists to keep small. */
export function runCascade(ii, stride, top, left, learners, stages) {
  let s = 0;
  let used = 0;
  let next = 0;
  for (let st = 0; st < stages.length; st++) {
    for (let i = next; i < stages[st].upto; i++) {
      const L = learners[i];
      s += L.alpha * stumpVote(featureValue(ii, stride, top, left, L.rects), L);
      used += 1;
    }
    next = stages[st].upto;
    if (s <= stages[st].threshold) return { survived: false, stage: st, used, score: s };
  }
  return { survived: true, stage: stages.length, used, score: s };
}

/* Every window of an image, scanned. */
export function scanImage(gray, w, h, win, learners, stages) {
  const ii = integralImage(gray, w, h);
  const stride = w + 1;
  const survivors = [];
  const perStage = new Array(stages.length + 1).fill(0);
  let used = 0;
  let n = 0;
  for (let top = 0; top + win <= h; top++) {
    for (let left = 0; left + win <= w; left++) {
      const r = runCascade(ii, stride, top, left, learners, stages);
      used += r.used;
      n += 1;
      perStage[r.stage] += 1;
      if (r.survived) survivors.push({ top, left, score: r.score });
    }
  }
  return { survivors, perStage, meanUsed: used / n, nWindows: n };
}
