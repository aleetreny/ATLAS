/* Entry point.
 *
 * The guards here are unusually strong, and that is the point of the article:
 * every claim on this page is a count or an exact answer, so the browser can
 * recompute all of it and demand agreement rather than closeness. Corners,
 * pivots, node counts and objective values are integers or exact rationals, so
 * a tolerance of 1e-9 is not a guess, it is the arithmetic of doubles.
 */
import { initDialWidget } from './dial-widget.js';
import { initSimplexScrolly } from './scrolly-simplex.js';
import { initKmWidget } from './km-widget.js';
import { initDualWidget } from './dual-widget.js';
import { initBbWidget } from './bb-widget.js';
import { initProse } from './prose.js';
import { enumerateVertices, ring, simplex, kleeMinty, branchAndBound, knapsackLP } from './lpkit.js';

function renderMath() {
  if (typeof renderMathInElement !== 'function') return;
  renderMathInElement(document.body, {
    delimiters: [
      { left: '$$', right: '$$', display: true },
      { left: '\\(', right: '\\)', display: false },
      { left: '$', right: '$', display: false },
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

function check(label, ok, detail) {
  if (!ok) console.warn(`[linear-programming] ${label}: ${detail}`);
  return ok;
}

function runGuards(data, deep) {
  const P = data.polytope;
  const A = P.rows.map((r) => r.a).concat([[-1, 0], [0, -1]]);
  const b = P.rows.map((r) => r.b).concat([0, 0]);

  /* 1. the corners, one by one against the file */
  const mine = ring(enumerateVertices(A, b).vertices);
  const theirs = P.vertices;
  let worstCorner = Infinity;
  const matched = theirs.every((t) => {
    const d = Math.min(...mine.map((m) => Math.hypot(m.x - t.x, m.y - t.y)));
    worstCorner = Math.min(worstCorner, d);
    return d < 1e-9;
  });
  check('the corners', matched && mine.length === theirs.length,
    `the browser finds ${mine.length} corners and the file has ${theirs.length}`);

  /* 2. the simplex path, corner by corner, and the objective value */
  const run = simplex(P.profit, P.rows.map((r) => r.a), P.rows.map((r) => r.b));
  const S = data.simplex;
  let worstPath = 0;
  S.path.forEach((p, i) => {
    if (!run.path[i]) { worstPath = Infinity; return; }
    worstPath = Math.max(worstPath, Math.abs(run.path[i][0] - p.x), Math.abs(run.path[i][1] - p.y));
  });
  check('the simplex walk', run.pivots === S.pivots && worstPath < 1e-9
    && Math.abs(run.value - S.value) < 1e-9,
  `${run.pivots} pivots here against ${S.pivots} in the file, worst corner off by ${worstPath}`);

  /* 3. the cube: 2^n - 1 pivots with the steepest rule, recomputed */
  let kmBad = 0;
  const sizes = deep ? S.klee_minty : S.klee_minty.slice(0, 5);
  sizes.forEach((row) => {
    const { c, A: Ak, b: bk } = kleeMinty(row.n);
    const d = simplex(c, Ak, bk, { rule: 'dantzig' });
    const bl = simplex(c, Ak, bk, { rule: 'bland' });
    if (d.pivots !== row.dantzig || d.pivots !== 2 ** row.n - 1 || bl.pivots !== row.bland) {
      kmBad += 1;
    }
  });
  check('the cube', kmBad === 0,
    `${kmBad} of ${sizes.length} sizes disagree with the file or with 2^n - 1`);

  /* 4. the shadow prices, from re-solving rather than from the dual */
  const b0 = P.rows.map((r) => r.b);
  const Aub = P.rows.map((r) => r.a);
  let worstShadow = 0;
  data.duals.rows.forEach((row, i) => {
    const h = 1e-6;
    const bump = (d) => {
      const bb = b0.slice();
      bb[i] = b0[i] + d;
      return simplex(P.profit, Aub, bb).value;
    };
    worstShadow = Math.max(worstShadow, Math.abs((bump(h) - bump(-h)) / (2 * h) - row.shadow));
  });
  check('the shadow prices', worstShadow < 1e-5,
    `re-solving disagrees with the dual by ${worstShadow.toExponential(2)}`);

  /* 5. branch and bound: the node counts, which are the article's claim */
  const M = data.demo;
  const withB = branchAndBound(M.weights, M.values, M.capacity, { useBound: true });
  const noB = branchAndBound(M.weights, M.values, M.capacity, { useBound: false });
  const lp = knapsackLP(M.weights, M.values, M.capacity);
  check('branch and bound', withB.nodes === M.bb_nodes && noB.nodes === M.bb_nodes_no_bound
    && Math.abs(withB.best - M.optimum) < 1e-9,
  `${withB.nodes} nodes here against ${M.bb_nodes} in the file, `
    + `${noB.nodes} against ${M.bb_nodes_no_bound}, best bag ${withB.best} against ${M.optimum}`);
  check('the relaxation', Math.abs(lp.value - M.lp_bound) < 1e-3,
    `${lp.value} here against ${M.lp_bound} in the file`);
  check('the ceiling', lp.value >= M.optimum - 1e-9,
    `the relaxation ${lp.value} is below the integer optimum ${M.optimum}, which is impossible`);

  /* 6. the dial: the switch angles are the edge normals, to the last decimal */
  const R = ring(enumerateVertices(A, b).vertices);
  const winner = (deg) => {
    const th = (deg * Math.PI) / 180;
    let bestJ = 0;
    let bestS = -Infinity;
    R.forEach((p, j) => {
      const s = p.x * Math.cos(th) + p.y * Math.sin(th);
      if (s > bestS) { bestS = s; bestJ = j; }
    });
    return bestJ;
  };
  const owners = new Set();
  for (let k = 0; k < 720; k++) owners.add(winner((360 * k) / 720));
  check('the dial', owners.size === data.dial.distinct_owners,
    `${owners.size} corners win somewhere in the browser and ${data.dial.distinct_owners} in the file`);

  /* 7. nothing may read undefined or NaN */
  const empty = [];
  document.querySelectorAll('p.body-text, .widget-readout, .atlas-table td, .gen-note, svg text')
    .forEach((node) => {
      const t = node.textContent || '';
      if (/\bundefined\b|\bNaN\b|\[object Object\]/.test(t)) {
        empty.push(node.id || node.className || node.tagName);
      }
    });
  check('composed text', empty.length === 0, `undefined or NaN appears in: ${empty.join(', ')}`);

  console.info('[linear-programming] load-time checks complete: '
    + `${mine.length} corners, ${run.pivots} pivots, ${withB.nodes} nodes, `
    + `shadow prices within ${worstShadow.toExponential(1)}`
    + `${deep ? '' : ' (spot check; window.__atlasCheck(true) runs every cube size too)'}`);
  return { corners: mine.length, pivots: run.pivots, nodes: withB.nodes, worstShadow };
}

async function boot() {
  renderMath();
  const data = await fetch('./data/lp.json').then((res) => res.json());

  safely('prose', () => initProse(data));
  safely('the dial', () => initDialWidget(data));
  safely('the simplex walk', () => initSimplexScrolly(data));
  safely('the cube', () => initKmWidget(data));
  safely('shadow prices', () => initDualWidget(data));
  safely('branch and bound', () => initBbWidget(data));

  renderMath();

  window.__atlasCheck = (deep = false) => safely('guards', () => runGuards(data, deep));
  const guards = () => window.__atlasCheck(false);
  if ('requestIdleCallback' in window) requestIdleCallback(guards, { timeout: 5000 });
  else setTimeout(guards, 1000);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
