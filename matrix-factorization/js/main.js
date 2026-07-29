/* Entry point.
 *
 * The guard here is the article's own argument run twice. The rotation
 * invariance is exact algebra, so the browser can check it on the vectors it
 * was sent, at a tolerance that is float noise rather than a tolerance anybody
 * chose; and the identity with the directions article is checked against that
 * article's published file, fetched at load time rather than copied in, so it
 * cannot drift.
 */
import { initHolesWidget } from './holes-widget.js';
import { initSweepWidget } from './sweep-widget.js';
import { initRotWidget } from './rot-widget.js';
import { initNbWidget } from './nb-widget.js';
import { initProse } from './prose.js';

let PROSE_IDS = [];

function renderMath() {
  if (typeof renderMathInElement !== 'function') return;
  renderMathInElement(document.body, {
    delimiters: [
      { left: '$$', right: '$$', display: true },
      { left: '\\(', right: '\\)', display: false },
    ],
    throwOnError: false,
  });
}

function safely(name, fn) {
  try {
    return fn();
  } catch (err) {
    console.error(`widget "${name}" failed to initialise:`, err);
    return null;
  }
}

function check(data, published, loud = false) {
  const problems = [];
  const fail = (label, detail) => problems.push(`${label}: ${detail}`);

  /* 1. the rotation is exact, so this is a real guard and not a tolerance
     negotiation: an orthogonal change of coordinates cannot move a dot
     product by more than the float noise of the sum. */
  const V = data.widget.full;
  const th = 0.7;
  const c = Math.cos(th);
  const s = Math.sin(th);
  const R = V.map((row) => {
    const out = row.slice();
    out[0] = c * row[0] - s * row[1];
    out[1] = s * row[0] + c * row[1];
    return out;
  });
  let worst = 0;
  for (let i = 0; i < 40; i++) {
    for (let j = i + 1; j < 40; j++) {
      let a = 0;
      let b = 0;
      for (let k = 0; k < V[i].length; k++) { a += V[i][k] * V[j][k]; b += R[i][k] * R[j][k]; }
      worst = Math.max(worst, Math.abs(a - b));
    }
  }
  if (!(worst < 1e-9)) {
    fail('the rotation', `a score moved by ${worst.toExponential(2)} under a rotation, which is `
      + 'algebraically impossible, so the vectors are not being read as vectors');
  }

  /* 2. the identity with the directions article, checked against that
     article's own file rather than against a number copied into this one */
  if (published) {
    const ref = {};
    published.atoms.errors.forEach((row) => { ref[row.rank] = row.pca_unquantised; });
    let gap = 0;
    data.digits.complete.forEach((row) => {
      if (ref[row.rank] !== undefined) gap = Math.max(gap, Math.abs(row.svd - ref[row.rank]));
    });
    /* the published file carries eight decimals, so 1e-7 is the floor of this
       comparison and anything tighter would fire on rounding */
    if (!(gap < 1e-6)) {
      fail('the identity with the directions article',
        `the truncated SVD here and the PCA published there differ by ${gap.toExponential(2)}`);
    }
  } else {
    fail('the identity with the directions article', 'its data file could not be read');
  }

  /* 3. the sweeps have to contain the numbers the prose quotes */
  const bestRank = data.sweeps.rank.reduce((a, x) => (x.test < a.test ? x : a));
  if (!(data.ratings.rmse.als <= bestRank.test * 1.02)) {
    fail('the headline fit', `the headline ALS run scores ${data.ratings.rmse.als} while the best `
      + `of the rank sweep is ${bestRank.test}, which is more than the extra passes can explain`);
  }

  /* 4. the composed text */
  const bad = [];
  document.querySelectorAll('p.body-text, .widget-readout, .atlas-table td, .rank-table td,'
    + ' .widget-note, svg text').forEach((node) => {
    const t = node.textContent || '';
    if (/\bundefined\b|\bNaN\b|\[object Object\]|\$\{/.test(t)) bad.push(node.id || node.className);
  });
  if (bad.length) fail('composed text', `undefined or NaN appears in: ${bad.join(', ')}`);

  if (problems.length) problems.forEach((p) => console.warn(`[matrix-factorization] ${p}`));
  else if (loud) {
    console.log('[matrix-factorization] the rotation moves nothing, the truncated SVD reproduces '
      + 'the PCA the directions article published, and the sweeps contain the headline');
  }
  return problems;
}

async function boot() {
  renderMath();
  const data = await fetch('./data/mf.json').then((r) => r.json());
  const published = await fetch('../directions/data/directions.json')
    .then((r) => r.json()).catch(() => null);

  PROSE_IDS = safely('the numbered sentences', () => initProse(data)) || [];
  safely('the hole', () => initHolesWidget(data));
  safely('the sweeps', () => initSweepWidget(data));
  safely('the rotation', () => initRotWidget(data));
  safely('the neighbours', () => initNbWidget(data));
  renderMath();

  const guards = () => safely('guards', () => check(data, published, true));
  if ('requestIdleCallback' in window) requestIdleCallback(guards, { timeout: 5000 });
  else setTimeout(guards, 800);

  window.__atlasCheck = (loud) => check(data, published, loud);
  window.__atlasProseIds = PROSE_IDS;
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
