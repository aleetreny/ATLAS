/* The rotation test, run live in the browser.
 *
 * This is the sharpest single argument in Grinsztajn et al. (2022) and it is
 * cheap enough to run as you scroll. Take a target whose structure is aligned
 * with the feature axes, which is what most spreadsheet columns look like.
 * Rotate the feature space. No information is created or destroyed: the rotation
 * is invertible and distances between points are unchanged, so any model that
 * only cares about distance is looking at exactly the same problem. A tree is
 * not such a model. It can only cut perpendicular to an axis, so after the
 * rotation it has to approximate every diagonal boundary with a staircase.
 *
 * Both models here are written from scratch in this file: a greedy CART on gini
 * and a k-nearest-neighbour vote. Nothing is precomputed, so the numbers in the
 * score panel are measured on your machine as you read.
 */
import { scrolly } from '../../assets/js/scrolly.js';

const N_TRAIN = 420;
const N_TEST = 2500;
const ANGLE = 35 * (Math.PI / 180);
const K = 15;
const DEPTH = 6;
const BLOCKS = 3; // the quilt is BLOCKS x BLOCKS
const LIM = 1.45; // display domain, wide enough to hold the rotated square

const C0 = 'var(--primary)';
const C1 = 'var(--smile)';

/* ------------------------------------------------------------------ data */
function mulberry32(a) {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* A checkerboard of axis-aligned blocks: the label depends on which band of
 * feature 1 and which band of feature 2 a point falls in, and on nothing else.
 * That is a caricature of tabular structure, and a caricature is the point. */
function label(x0, x1) {
  const i = Math.min(BLOCKS - 1, Math.floor(((x0 + 1) / 2) * BLOCKS));
  const j = Math.min(BLOCKS - 1, Math.floor(((x1 + 1) / 2) * BLOCKS));
  return (i + j) % 2;
}

function sample(n, seed) {
  const rand = mulberry32(seed);
  const X = [];
  const y = [];
  for (let i = 0; i < n; i++) {
    const a = rand() * 2 - 1;
    const b = rand() * 2 - 1;
    X.push([a, b]);
    y.push(label(a, b));
  }
  return { X, y };
}

const rotate = (X, t) =>
  X.map(([a, b]) => [a * Math.cos(t) - b * Math.sin(t), a * Math.sin(t) + b * Math.cos(t)]);

/* ------------------------------------------------------------------ CART */
function fitTree(X, y, maxDepth = DEPTH, minLeaf = 6) {
  const build = (idx, depth) => {
    const n = idx.length;
    let ones = 0;
    for (const i of idx) ones += y[i];
    const p = ones / n;
    const node = { p, value: p >= 0.5 ? 1 : 0, n };
    if (depth >= maxDepth || n < 2 * minLeaf || ones === 0 || ones === n) return node;

    let best = null;
    for (let f = 0; f < 2; f++) {
      const order = [...idx].sort((a, b) => X[a][f] - X[b][f]);
      let left1 = 0;
      for (let k = 0; k < n - 1; k++) {
        left1 += y[order[k]];
        const nl = k + 1;
        const nr = n - nl;
        if (nl < minLeaf || nr < minLeaf) continue;
        if (X[order[k]][f] === X[order[k + 1]][f]) continue;
        const pl = left1 / nl;
        const pr = (ones - left1) / nr;
        const gini = (nl * 2 * pl * (1 - pl) + nr * 2 * pr * (1 - pr)) / n;
        if (!best || gini < best.gini) {
          best = { gini, f, thr: (X[order[k]][f] + X[order[k + 1]][f]) / 2 };
        }
      }
    }
    if (!best) return node;

    const li = [];
    const ri = [];
    for (const i of idx) (X[i][best.f] <= best.thr ? li : ri).push(i);
    if (!li.length || !ri.length) return node;
    node.split = best;
    node.left = build(li, depth + 1);
    node.right = build(ri, depth + 1);
    return node;
  };
  return build(
    Array.from({ length: X.length }, (_, i) => i),
    0
  );
}

function treePredict(root, x) {
  let node = root;
  while (node.split) node = x[node.split.f] <= node.split.thr ? node.left : node.right;
  return node.value;
}

/* A tree's decision surface IS a set of rectangles, so read them straight off
 * the structure rather than sampling a pixel grid. This is the whole point of
 * the section made literal: everything it can express is a box. */
function leafRects(node, box = [-LIM, -LIM, LIM, LIM], out = []) {
  if (!node.split) {
    out.push({ x0: box[0], y0: box[1], x1: box[2], y1: box[3], value: node.value, p: node.p });
    return out;
  }
  const { f, thr } = node.split;
  const lo = [...box];
  const hi = [...box];
  if (f === 0) {
    lo[2] = Math.min(box[2], thr);
    hi[0] = Math.max(box[0], thr);
  } else {
    lo[3] = Math.min(box[3], thr);
    hi[1] = Math.max(box[1], thr);
  }
  leafRects(node.left, lo, out);
  leafRects(node.right, hi, out);
  return out;
}

/* ------------------------------------------------------------------- kNN */
function knnField(X, y, x, k = K) {
  /* Partial selection beats a full sort: k is 15 and the training set is 420,
   * and this runs a few thousand times per redraw. */
  const bd = new Float64Array(k).fill(Infinity);
  const bi = new Int32Array(k).fill(-1);
  for (let i = 0; i < X.length; i++) {
    const dx = X[i][0] - x[0];
    const dy = X[i][1] - x[1];
    const d = dx * dx + dy * dy;
    if (d >= bd[k - 1]) continue;
    let j = k - 1;
    while (j > 0 && bd[j - 1] > d) {
      bd[j] = bd[j - 1];
      bi[j] = bi[j - 1];
      j--;
    }
    bd[j] = d;
    bi[j] = i;
  }
  let s = 0;
  for (let j = 0; j < k; j++) s += y[bi[j]];
  return s / k;
}

const errRate = (yTrue, yPred) => {
  let bad = 0;
  for (let i = 0; i < yTrue.length; i++) if (yTrue[i] !== yPred[i]) bad++;
  return bad / yTrue.length;
};

/* ------------------------------------------------------------------- view */
export function initScrollyRotation() {
  const W = 700;
  const H = 600;
  const M = { top: 46, right: 186, bottom: 44, left: 44 };
  const w = W - M.left - M.right;
  const h = H - M.top - M.bottom;

  const svg = d3
    .select('#rot-chart')
    .append('svg')
    .attr('viewBox', `0 0 ${W} ${H}`)
    .attr('preserveAspectRatio', 'xMidYMid meet');
  const g = svg.append('g').attr('transform', `translate(${M.left},${M.top})`);

  const side = Math.min(w, h);
  const x = d3.scaleLinear().domain([-LIM, LIM]).range([0, side]);
  const y = d3.scaleLinear().domain([-LIM, LIM]).range([side, 0]);

  g.append('rect')
    .attr('width', side)
    .attr('height', side)
    .attr('fill', 'var(--white)')
    .attr('stroke', 'var(--squidink)')
    .attr('stroke-width', 1.5);

  const clipId = 'rot-clip';
  svg
    .append('clipPath')
    .attr('id', clipId)
    .append('rect')
    .attr('width', side)
    .attr('height', side);

  const surface = g.append('g').attr('clip-path', `url(#${clipId})`);
  const truth = g.append('g').attr('clip-path', `url(#${clipId})`).attr('opacity', 0);
  /* Two ways to show the truth. Solid, when it is the subject. Outline only,
   * when a fitted surface is underneath and the question is how well the two
   * agree: a wash on top of a wash just makes a third colour and hides the
   * comparison the step exists to make. */
  const setTruth = (mode, dur = 300) => {
    truth.attr('opacity', 1);
    truth
      .selectAll('rect')
      .transition('fade')
      .duration(dur)
      .attr('fill-opacity', mode === 'solid' ? 0.16 : 0)
      .attr('stroke-opacity', mode === 'solid' ? 0.25 : 0.6);
  };
  const dots = g.append('g').attr('clip-path', `url(#${clipId})`);
  const caption = g
    .append('text')
    .attr('class', 'chart-title')
    .attr('x', side / 2)
    .attr('y', -18)
    .attr('text-anchor', 'middle');

  /* the true quilt, drawn as real blocks so the rotation can be shown by
   * rotating the group rather than by redrawing anything */
  const step = 2 / BLOCKS;
  const cells = [];
  for (let i = 0; i < BLOCKS; i++) {
    for (let j = 0; j < BLOCKS; j++) {
      cells.push({ a: -1 + i * step, b: -1 + j * step, c: (i + j) % 2 });
    }
  }
  const cx = x(0);
  const cy = y(0);
  truth
    .selectAll('rect')
    .data(cells)
    .join('rect')
    .attr('x', (d) => x(d.a))
    .attr('y', (d) => y(d.b + step))
    .attr('width', x(step) - x(0))
    .attr('height', y(0) - y(step))
    .attr('fill', (d) => (d.c ? C1 : C0))
    .attr('fill-opacity', 0.16)
    .attr('stroke', 'var(--squidink)')
    .attr('stroke-width', 1.2)
    .attr('stroke-opacity', 0.25);

  /* ---------------------------------------------------------- the numbers */
  const train = sample(N_TRAIN, 7);
  const test = sample(N_TEST, 99);
  const trainR = { X: rotate(train.X, ANGLE), y: train.y };
  const testR = { X: rotate(test.X, ANGLE), y: test.y };

  const memo = {};
  const measure = (key, fn) => (key in memo ? memo[key] : (memo[key] = fn()));

  const treePlain = () => measure('tp', () => fitTree(train.X, train.y));
  const treeRot = () => measure('tr', () => fitTree(trainR.X, trainR.y));
  /* The obvious objection to the step above is that depth 6 handicapped the
   * rotated tree, so answer it in the demo rather than in a footnote. Depth 14
   * is past the point where more makes any difference to either fit. */
  const treeRotDeep = () => measure('trd', () => fitTree(trainR.X, trainR.y, 14));
  const errTreePlain = () =>
    measure('etp', () => errRate(test.y, test.X.map((p) => treePredict(treePlain(), p))));
  const errTreeRot = () =>
    measure('etr', () => errRate(testR.y, testR.X.map((p) => treePredict(treeRot(), p))));
  const errTreeRotDeep = () =>
    measure('etrd', () => errRate(testR.y, testR.X.map((p) => treePredict(treeRotDeep(), p))));
  const errKnnPlain = () =>
    measure('ekp', () =>
      errRate(test.y, test.X.map((p) => (knnField(train.X, train.y, p) >= 0.5 ? 1 : 0)))
    );
  const errKnnRot = () =>
    measure('ekr', () =>
      errRate(testR.y, testR.X.map((p) => (knnField(trainR.X, trainR.y, p) >= 0.5 ? 1 : 0)))
    );

  /* ------------------------------------------------------- the score panel */
  const panel = g.append('g').attr('transform', `translate(${side + 26},4)`);
  panel
    .append('text')
    .attr('class', 'chart-annotation')
    .attr('x', 0)
    .attr('y', 0)
    .style('font-size', '0.62rem')
    .text('held-out error');

  const countLeaves = (t) => leafRects(t).length;
  const ROWS = [
    { key: 'tree-plain', label: 'tree, aligned', get: errTreePlain, note: () => `${countLeaves(treePlain())} leaves`, colour: 'var(--squidink)' },
    { key: 'tree-rot', label: 'tree, rotated', get: errTreeRot, note: () => `${countLeaves(treeRot())} leaves`, colour: 'var(--cosmos)' },
    { key: 'tree-deep', label: 'same, no depth cap', get: errTreeRotDeep, note: () => `${countLeaves(treeRotDeep())} leaves`, colour: 'var(--cosmos)' },
    { key: 'knn-plain', label: `${K}-nn, aligned`, get: errKnnPlain, note: () => '', colour: 'var(--squidink)' },
    { key: 'knn-rot', label: `${K}-nn, rotated`, get: errKnnRot, note: () => '', colour: 'var(--seablue)' },
  ];
  const rowG = panel
    .selectAll('g.row')
    .data(ROWS)
    .join('g')
    .attr('class', 'row')
    .attr('transform', (d, i) => `translate(0,${24 + i * 44})`)
    .attr('opacity', 0);
  rowG
    .append('text')
    .attr('class', 'chart-annotation')
    .attr('x', 0)
    .attr('y', 0)
    .style('font-size', '0.6rem')
    .style('text-transform', 'none')
    .text((d) => d.label);
  const rowVal = rowG
    .append('text')
    .attr('x', 0)
    .attr('y', 18)
    .attr('fill', (d) => d.colour)
    .style('font-family', 'var(--font-heavy)')
    .style('font-size', '15px');
  const rowNote = rowG
    .append('text')
    .attr('class', 'chart-annotation')
    .attr('x', 0)
    .attr('y', 31)
    .style('font-size', '0.55rem')
    .style('text-transform', 'none')
    .attr('opacity', 0.7);

  let shown = new Set();
  function showRows(keys) {
    shown = new Set(keys);
    rowG.transition().duration(280).attr('opacity', (d) => (shown.has(d.key) ? 1 : 0));
    rowVal.each(function (d) {
      if (shown.has(d.key)) d3.select(this).text(`${(d.get() * 100).toFixed(1)}%`);
    });
    rowNote.each(function (d) {
      if (shown.has(d.key)) d3.select(this).text(d.note());
    });
  }

  /* ------------------------------------------------------------ rendering */
  let pointsAt = train.X; // where the dots currently sit
  function drawDots(coords, dur = 0) {
    const sel = dots
      .selectAll('circle')
      .data(train.y.map((c, i) => ({ c, p: coords[i] })))
      .join((enter) =>
        enter
          .append('circle')
          .attr('r', 3.6)
          .attr('cx', (d) => x(d.p[0]))
          .attr('cy', (d) => y(d.p[1]))
          .attr('stroke', 'var(--squidink)')
          .attr('stroke-width', 0.7)
      )
      .attr('fill', (d) => (d.c ? C1 : C0));
    (dur ? sel.transition().duration(dur) : sel)
      .attr('cx', (d) => x(d.p[0]))
      .attr('cy', (d) => y(d.p[1]));
    pointsAt = coords;
  }

  /* Both surfaces are drawn in the same visual language: one wash per class and
   * a boundary. Only the shape of the boundary differs, which is the entire
   * argument of this section, so nothing else should. */
  const WASH = 0.26;

  function drawTree(root) {
    const rects = leafRects(root);
    surface.selectAll('.knn').remove();
    surface
      .selectAll('rect.leaf')
      .data(rects)
      .join('rect')
      .attr('class', 'leaf')
      .attr('x', (d) => x(d.x0))
      .attr('y', (d) => y(d.y1))
      .attr('width', (d) => Math.max(0, x(d.x1) - x(d.x0)))
      .attr('height', (d) => Math.max(0, y(d.y0) - y(d.y1)))
      .attr('fill', (d) => (d.value ? C1 : C0))
      /* faded where the leaf is mixed, solid where it is pure: a leaf that is
       * 55% orange should not look as confident as one that is 100% */
      .attr('fill-opacity', (d) => WASH * (0.45 + 1.1 * Math.abs(d.p - 0.5)))
      .attr('stroke', 'var(--squidink)')
      .attr('stroke-width', 0.6)
      .attr('stroke-opacity', 0.45);
  }

  function drawKnn(X, yv) {
    const n = 64;
    const vals = new Array(n * n);
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        const a = -LIM + ((i + 0.5) / n) * 2 * LIM;
        const b = LIM - ((j + 0.5) / n) * 2 * LIM;
        vals[j * n + i] = knnField(X, yv, [a, b]);
      }
    }
    const region = d3.contours().size([n, n]).thresholds([0.5])(vals)[0];
    const sx = side / n;
    const d = d3.geoPath()(region);
    surface.selectAll('rect.leaf').remove();

    /* Everything the vote calls class 0, then the class-1 region punched out of
     * it in white and refilled, so the two washes stay their own colours
     * instead of blending into a third. */
    surface
      .selectAll('rect.knn-base')
      .data([0])
      .join('rect')
      .attr('class', 'knn knn-base')
      .attr('width', side)
      .attr('height', side)
      .attr('fill', C0)
      .attr('fill-opacity', WASH);
    surface
      .selectAll('path.knn-layer')
      .data([
        { fill: 'var(--white)', op: 1, stroke: 0 },
        { fill: C1, op: WASH, stroke: 0 },
        { fill: 'none', op: 1, stroke: 2.2 / sx },
      ])
      .join('path')
      .attr('class', 'knn knn-layer contour')
      .attr('transform', `scale(${sx})`)
      .attr('d', d)
      .attr('fill', (l) => l.fill)
      .attr('fill-opacity', (l) => l.op)
      .attr('stroke', 'var(--squidink)')
      .attr('stroke-width', (l) => l.stroke)
      .attr('stroke-opacity', 0.8);
  }

  const clearSurface = () => surface.selectAll('*').remove();

  /* ---------------------------------------------------------------- steps */
  let rotated = false;
  function setRotated(want, animate) {
    if (rotated === want) return;
    rotated = want;
    truth
      .transition('spin')
      .duration(animate ? 750 : 0)
      .attr('transform', want ? `rotate(${-(ANGLE * 180) / Math.PI},${cx},${cy})` : null);
    drawDots(want ? trainR.X : train.X, animate ? 750 : 0);
  }

  const handlers = {
    0: () => {
      setRotated(false, false);
      clearSurface();
      setTruth('solid');
      caption.text('the truth: nine blocks, cut along the axes');
      showRows([]);
    },
    1: () => {
      setRotated(false, false);
      setTruth('outline');
      drawTree(treePlain());
      caption.text('a tree, depth 6');
      showRows(['tree-plain']);
    },
    2: () => {
      setRotated(true, true);
      setTruth('solid');
      clearSurface();
      caption.text('the same points, seen from 35 degrees off');
      showRows(['tree-plain']);
    },
    3: () => {
      setRotated(true, false);
      setTruth('outline');
      drawTree(treeRot());
      caption.text('the same tree, refitted');
      showRows(['tree-plain', 'tree-rot']);
    },
    4: () => {
      setRotated(true, false);
      setTruth('outline');
      drawTree(treeRotDeep());
      caption.text('the same tree with the depth limit removed');
      showRows(['tree-plain', 'tree-rot', 'tree-deep']);
    },
    5: () => {
      setRotated(true, false);
      setTruth('outline');
      drawKnn(trainR.X, trainR.y);
      caption.text(`${K} nearest neighbours, rotated`);
      showRows(['tree-plain', 'tree-rot', 'tree-deep', 'knn-plain', 'knn-rot']);
    },
  };

  drawDots(train.X);
  handlers[0]();
  return scrolly('#rot-scrolly .step', handlers);
}
