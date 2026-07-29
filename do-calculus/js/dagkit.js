/* The model, in the browser, computed rather than estimated.
 *
 * Everything here is a finite sum over the 2^k assignments of k binary
 * variables. That is why this article's guard can demand 1e-12 where the rest
 * of the atlas has to settle for a tolerance derived from how many decimals
 * something was written with: there is no sampling anywhere, so the only
 * difference between the two sides is the order of a few dozen additions.
 */

/* All 2^k assignments, as arrays of zeros and ones. */
export function worlds(k) {
  const out = [];
  for (let m = 0; m < (1 << k); m++) {
    const w = new Array(k);
    for (let i = 0; i < k; i++) w[i] = (m >> (k - 1 - i)) & 1;
    out.push(w);
  }
  return out;
}

/* The probability of one assignment, with the factors of intervened variables
 * crossed out. That crossing out IS the intervention: everything else about
 * the model stays exactly as it was. */
export function rowProb(model, w, doMap) {
  let p = 1;
  for (let i = 0; i < model.parents.length; i++) {
    if (doMap && Object.prototype.hasOwnProperty.call(doMap, i)) {
      if (w[i] !== doMap[i]) return 0;
      continue;
    }
    let idx = 0;
    for (const pa of model.parents[i]) idx = idx * 2 + w[pa];
    const pi = model.cpt[i][idx];
    p *= w[i] === 1 ? pi : 1 - pi;
  }
  return p;
}

export function jointProb(model, ws, doMap = null) {
  return ws.map((w) => rowProb(model, w, doMap));
}

/* P(Y = 1 | do(X = value)). */
export function doProb(model, ws, xNode, value, yNode) {
  const j = jointProb(model, ws, { [xNode]: value });
  let tot = 0;
  let num = 0;
  ws.forEach((w, i) => {
    tot += j[i];
    if (w[yNode] === 1) num += j[i];
  });
  return tot > 0 ? num / tot : NaN;
}

/* The adjustment formula over a set S, from the observational joint. */
export function adjust(model, ws, xNode, yNode, S) {
  const j = jointProb(model, ws);
  const k = S.length;
  let acc = 0;
  for (let m = 0; m < (1 << k); m++) {
    const vals = S.map((_, t) => (m >> (k - 1 - t)) & 1);
    let ps = 0;
    const inS = ws.map((w, i) => {
      const ok = S.every((s, t) => w[s] === vals[t]);
      if (ok) ps += j[i];
      return ok;
    });
    if (ps <= 1e-15) continue;
    const arm = [0, 1].map((xv) => {
      let num = 0;
      let den = 0;
      ws.forEach((w, i) => {
        if (!inS[i] || w[xNode] !== xv) return;
        den += j[i];
        if (w[yNode] === 1) num += j[i];
      });
      return den > 1e-15 ? num / den : NaN;
    });
    if (!Number.isFinite(arm[0]) || !Number.isFinite(arm[1])) return NaN;
    acc += ps * (arm[1] - arm[0]);
  }
  return acc;
}

/* Descendants of a node, for drawing and for the criterion. */
export function children(model, i) {
  const out = [];
  model.parents.forEach((pa, j) => { if (pa.includes(i)) out.push(j); });
  return out;
}
