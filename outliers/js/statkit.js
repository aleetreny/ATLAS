/* The statistics this article is about, written once and used by every widget,
 * by the guards and by the prose. No DOM in here on purpose: the same file can
 * be imported by node and checked against the generator without a browser.
 *
 * Two conventions matter enough to write down, because both have cost this repo
 * a wrong number before:
 *   - the spread is the SAMPLE standard deviation (divide by n-1), because that
 *     is what every textbook z-score rule and every library means by `sd`;
 *   - quantiles are linear interpolation between order statistics, which is
 *     numpy's default and R's type 7. Any other convention moves Tukey's fences
 *     between points and the browser and the generator stop agreeing.
 */

export function mean(v) {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i];
  return s / v.length;
}

/* Sample standard deviation, n-1 in the denominator. */
export function sd(v, m = null) {
  const mu = m === null ? mean(v) : m;
  let s = 0;
  for (let i = 0; i < v.length; i++) s += (v[i] - mu) ** 2;
  return Math.sqrt(s / (v.length - 1));
}

export function median(v) {
  return quantile(v, 0.5);
}

/* numpy's default: linear interpolation on (n-1)q. */
export function quantile(v, q) {
  const s = Float64Array.from(v).sort();
  const h = (s.length - 1) * q;
  const lo = Math.floor(h);
  const hi = Math.ceil(h);
  return s[lo] + (s[hi] - s[lo]) * (h - lo);
}

/* Median absolute deviation from the median. */
export function mad(v, med = null) {
  const m = med === null ? median(v) : med;
  return median(Array.from(v, (x) => Math.abs(x - m)));
}

export function zScores(v) {
  const m = mean(v);
  const s = sd(v, m);
  return { mean: m, sd: s, z: Array.from(v, (x) => (x - m) / s) };
}

/* Iglewicz and Hoaglin's modified z. The 0.6745 is the 0.75 quantile of the
 * standard normal, which makes MAD estimate the same thing as sd on gaussian
 * data, so this score and the one above live on one scale. */
export const MAD_CONST = 0.6745;
export function modifiedZ(v) {
  const med = median(v);
  const d = mad(v, med);
  return {
    median: med,
    mad: d,
    z: Array.from(v, (x) => (d === 0 ? (x === med ? 0 : Infinity) : (MAD_CONST * (x - med)) / d)),
  };
}

export function tukeyFences(v, whisker = 1.5) {
  const q1 = quantile(v, 0.25);
  const q3 = quantile(v, 0.75);
  const iqr = q3 - q1;
  return { q1, q3, iqr, lo: q1 - whisker * iqr, hi: q3 + whisker * iqr };
}

/* The ceiling on a z-score: put k of the n values at distance d and the rest
 * where they were, and d cancels out of the ratio entirely. Nothing about the
 * data survives into this number except n and k. */
export function zCeiling(n, k = 1) {
  return Math.sqrt(((n - 1) * (n - k)) / (n * k));
}

/* ------------------------------------------------------------ several columns */

/* Column means. */
export function colMeans(X) {
  const n = X.length;
  const p = X[0].length;
  const m = new Float64Array(p);
  for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) m[j] += X[i][j];
  for (let j = 0; j < p; j++) m[j] /= n;
  return m;
}

/* Sample covariance, n-1 again. Returned as an array of Float64Array rows. */
export function covariance(X, mu = null) {
  const n = X.length;
  const p = X[0].length;
  const m = mu === null ? colMeans(X) : mu;
  const S = Array.from({ length: p }, () => new Float64Array(p));
  for (let i = 0; i < n; i++) {
    for (let a = 0; a < p; a++) {
      const da = X[i][a] - m[a];
      for (let b = a; b < p; b++) S[a][b] += da * (X[i][b] - m[b]);
    }
  }
  for (let a = 0; a < p; a++) {
    for (let b = a; b < p; b++) {
      S[a][b] /= n - 1;
      S[b][a] = S[a][b];
    }
  }
  return S;
}

/* Cholesky, lower triangular. Returns null if the matrix is not positive
 * definite, which is the honest answer when there are fewer rows than columns
 * and is a case this article has to be able to report rather than crash on. */
export function cholesky(S) {
  const p = S.length;
  const L = Array.from({ length: p }, () => new Float64Array(p));
  for (let i = 0; i < p; i++) {
    for (let j = 0; j <= i; j++) {
      let s = S[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i === j) {
        if (!(s > 0)) return null;
        L[i][i] = Math.sqrt(s);
      } else {
        L[i][j] = s / L[j][j];
      }
    }
  }
  return L;
}

/* Solve S x = b through a Cholesky factor, which is how the squared distance
 * gets computed without ever forming an inverse: d^2 = (x-mu)' S^-1 (x-mu) is
 * the squared length of the forward substitution, and nothing else. */
export function cholSolveNorm(L, b) {
  const p = L.length;
  let s = 0;
  const y = new Float64Array(p);
  for (let i = 0; i < p; i++) {
    let t = b[i];
    for (let k = 0; k < i; k++) t -= L[i][k] * y[k];
    y[i] = t / L[i][i];
    s += y[i] * y[i];
  }
  return s;
}

/* Squared Mahalanobis distance of every row of X to (mu, S).
 * Pass rows = X to score the sample against its own estimate, which is the
 * whole subject of this page. */
export function mahalanobis(rows, mu, S) {
  const L = cholesky(S);
  if (!L) return null;
  const p = mu.length;
  const b = new Float64Array(p);
  return rows.map((x) => {
    for (let j = 0; j < p; j++) b[j] = x[j] - mu[j];
    return cholSolveNorm(L, b);
  });
}

/* The convenience version: fit and score in one call. */
export function selfMahalanobis(X) {
  const mu = colMeans(X);
  const S = covariance(X, mu);
  return { mu, S, d2: mahalanobis(X, mu, S) };
}

/* The ceiling in p dimensions, and it is the same algebra as the z one:
 * whatever the data, no row can score more than this against its own sample
 * mean and covariance. */
export function d2Ceiling(n) {
  return ((n - 1) ** 2) / n;
}

/* ------------------------------------------------------------------ the tails */

/* Regularised lower incomplete gamma P(s, x), by series below the diagonal and
 * continued fraction above it. Enough for a chi-squared tail on the degrees of
 * freedom this page uses, and it is checked at load time against the values the
 * generator got from scipy. */
function gammaln(x) {
  const c = [76.18009172947146, -86.50532032941677, 24.01409824083091,
    -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  let y = x;
  let t = x + 5.5;
  t -= (x + 0.5) * Math.log(t);
  let s = 1.000000000190015;
  for (let j = 0; j < 6; j++) s += c[j] / ++y;
  return -t + Math.log((2.5066282746310005 * s) / x);
}

function lowerGamma(s, x) {
  if (x < 0 || s <= 0) return NaN;
  if (x === 0) return 0;
  if (x < s + 1) {
    let ap = s;
    let sum = 1 / s;
    let del = sum;
    for (let i = 0; i < 500; i++) {
      ap += 1;
      del *= x / ap;
      sum += del;
      if (Math.abs(del) < Math.abs(sum) * 1e-15) break;
    }
    return sum * Math.exp(-x + s * Math.log(x) - gammaln(s));
  }
  let b = x + 1 - s;
  let c = 1e300;
  let d = 1 / b;
  let h = d;
  for (let i = 1; i < 500; i++) {
    const an = -i * (i - s);
    b += 2;
    d = an * d + b;
    if (Math.abs(d) < 1e-300) d = 1e-300;
    c = b + an / c;
    if (Math.abs(c) < 1e-300) c = 1e-300;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 1e-15) break;
  }
  return 1 - Math.exp(-x + s * Math.log(x) - gammaln(s)) * h;
}

/* P(chi^2_df <= x). */
export function chi2cdf(x, df) {
  return lowerGamma(df / 2, x / 2);
}

/* The cutoff, by bisection on the cdf. Slower than a table and impossible to
 * get subtly wrong, which is the trade this page wants. */
export function chi2ppf(p, df) {
  let lo = 0;
  let hi = Math.max(10, df * 8);
  while (chi2cdf(hi, df) < p) hi *= 2;
  for (let i = 0; i < 200; i++) {
    const mid = (lo + hi) / 2;
    if (chi2cdf(mid, df) < p) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
}

/* The standard normal tail, for the marginal rule the ellipse is compared
 * against. */
export function normCdf(x) {
  return 0.5 * (1 + erf(x / Math.SQRT2));
}

function erf(x) {
  const s = x < 0 ? -1 : 1;
  const a = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * a);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
    + 0.254829592) * t * Math.exp(-a * a);
  return s * y;
}

/* The 1 - alpha/2 point of the standard normal, by bisection for the same
 * reason as above. */
export function normPpf(p) {
  let lo = -10;
  let hi = 10;
  for (let i = 0; i < 200; i++) {
    const mid = (lo + hi) / 2;
    if (normCdf(mid) < p) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
}

/* ------------------------------------------------------------------- drawing */

/* The 2 by 2 ellipse {x : (x-mu)' S^-1 (x-mu) = c} as a closed polygon, which
 * is the only thing any of the pictures need. Built from the eigen-decomposition
 * because that is what makes the axes meaningful, and 2 by 2 symmetric
 * eigenvalues are a quadratic. */
export function ellipsePath(mu, S, c, steps = 128) {
  const [a, b, d] = [S[0][0], S[0][1], S[1][1]];
  const tr = a + d;
  const det = a * d - b * b;
  const disc = Math.sqrt(Math.max(0, (tr * tr) / 4 - det));
  const l1 = tr / 2 + disc;
  const l2 = tr / 2 - disc;
  /* the eigenvector for l1; if b is zero the axes are already aligned */
  let vx = 1;
  let vy = 0;
  if (Math.abs(b) > 1e-14) {
    vx = l1 - d;
    vy = b;
    const nrm = Math.hypot(vx, vy);
    vx /= nrm;
    vy /= nrm;
  }
  const r1 = Math.sqrt(Math.max(0, c * l1));
  const r2 = Math.sqrt(Math.max(0, c * l2));
  const pts = [];
  for (let i = 0; i <= steps; i++) {
    const t = (2 * Math.PI * i) / steps;
    const u = r1 * Math.cos(t);
    const v = r2 * Math.sin(t);
    pts.push([mu[0] + u * vx - v * vy, mu[1] + u * vy + v * vx]);
  }
  return pts;
}
