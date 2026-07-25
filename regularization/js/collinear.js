/* Lasso's known weakness, and ElasticNet's reason to exist.
 *
 * Two predictors carrying nearly the same information. Lasso must break the tie
 * and does so arbitrarily; ElasticNet adds a pinch of L2, which prefers to split
 * credit, and stops flapping around.
 *
 * Note on methodology, because the obvious approach does not actually show this.
 * Bootstrapping the shipped sample does NOT flip Lasso's choice: whichever twin
 * happens to correlate marginally better with y in THAT sample keeps winning in
 * every resample of it, since resampling preserves the accident. Measured over
 * 40 bootstraps the winner never changed once. What the textbook claim is really
 * about is repeating the EXPERIMENT, so this widget draws a genuinely fresh
 * sample from the same generating process each time. Then the coin flip is real.
 */
import { makeChart, drawAxes, axisLabels } from '../../assets/js/chart.js';
import { makeModel } from './mathkit.js';

const ALPHA = 0.15;
const L1_RATIO_EN = 0.35;
const TWIN_NOISE = 0.14;
const SIGNAL = 3.0;
const Y_NOISE = 1.0;

/* mulberry32 + Box-Muller: seeded, so any run of the article is reproducible */
function rng(seed) {
  let a = seed >>> 0;
  const unif = () => {
    a += 0x6d2b79f5;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return () => {
    const u = Math.max(unif(), 1e-12);
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * unif());
  };
}

/* One fresh run of the same experiment: two near-duplicate predictors that
 * genuinely share the signal, plus four features that are pure noise. */
function drawSample(seed, n, nNoise) {
  const g = rng(seed);
  const base = Array.from({ length: n }, () => g());
  const cols = [
    base.map((b) => b + TWIN_NOISE * g()),
    base.map((b) => b + TWIN_NOISE * g()),
  ];
  for (let k = 0; k < nNoise; k++) cols.push(Array.from({ length: n }, () => g()));

  // standardize every column: a size penalty is only fair on a common scale
  const X = [];
  const mu = cols.map((c) => c.reduce((s, v) => s + v, 0) / n);
  const sd = cols.map((c, j) => Math.sqrt(c.reduce((s, v) => s + (v - mu[j]) ** 2, 0) / n) || 1);
  for (let i = 0; i < n; i++) X.push(cols.map((c, j) => (c[i] - mu[j]) / sd[j]));

  const yRaw = base.map((b) => SIGNAL * b + Y_NOISE * g());
  const ym = yRaw.reduce((s, v) => s + v, 0) / n;
  return { X, y: yRaw.map((v) => v - ym) };
}

export function initCollinear(col) {
  const n = col.X.length;
  const p = col.labels.length;

  const { g, w, h } = makeChart('#collinear-chart', {
    width: 620, height: 340, margin: { top: 34, right: 24, bottom: 78, left: 62 },
  });

  const x0 = d3.scaleBand().domain(d3.range(p)).range([0, w]).padding(0.28);
  const x1 = d3.scaleBand().domain(['lasso', 'enet']).range([0, x0.bandwidth()]).padding(0.12);
  const y = d3.scaleLinear().domain([-0.4, 2.4]).range([h, 0]);

  drawAxes(g, x0, y, w, h);
  g.select('g.axis.x').selectAll('text')
    .attr('transform', 'rotate(-28)')
    .attr('text-anchor', 'end')
    .attr('dx', '-4')
    .text((d) => col.labels[d]);
  axisLabels(g, w, h, { y: 'coefficient' });

  g.append('line').attr('x1', 0).attr('x2', w).attr('y1', y(0)).attr('y2', y(0))
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.5);

  const barsG = g.append('g');
  const legend = g.append('g').attr('transform', `translate(${w - 190},-14)`);
  [['lasso', 'var(--cosmos)'], ['elasticnet', 'var(--aqua)']].forEach(([name, color], i) => {
    const row = legend.append('g').attr('transform', `translate(${i * 100},0)`);
    row.append('rect').attr('width', 12).attr('height', 12).attr('y', -10).attr('fill', color);
    row.append('text').attr('x', 17).attr('y', 0)
      .style('font-family', 'var(--font-mono)').style('font-size', '11.5px')
      .style('fill', 'var(--squidink)').text(name);
  });

  const statsEl = document.querySelector('#collinear-stats');
  const tallyEl = document.querySelector('#collinear-tally');
  const history = { lasso: [], enet: [], lassoWinA: 0, enetWinA: 0 };
  let run = 0;

  function solve(sample) {
    const m = makeModel(sample.X, sample.y);
    return { lasso: m.fit(ALPHA, 1), enet: m.fit(ALPHA, L1_RATIO_EN) };
  }

  function render(res, isFirst) {
    const data = [];
    for (let j = 0; j < p; j++) {
      data.push({ j, kind: 'lasso', v: res.lasso[j], color: 'var(--cosmos)' });
      data.push({ j, kind: 'enet', v: res.enet[j], color: 'var(--aqua)' });
    }
    barsG
      .selectAll('rect')
      .data(data)
      .join('rect')
      .attr('x', (d) => x0(d.j) + x1(d.kind))
      .attr('width', x1.bandwidth())
      .attr('fill', (d) => d.color)
      .transition()
      .duration(isFirst ? 0 : 420)
      .attr('y', (d) => (d.v >= 0 ? y(d.v) : y(0)))
      .attr('height', (d) => Math.abs(y(d.v) - y(0)));

    // how lopsided is the split between the two twins?
    const gap = (b) => {
      const tot = Math.abs(b[0]) + Math.abs(b[1]);
      return tot < 1e-9 ? 0 : Math.abs(Math.abs(b[0]) - Math.abs(b[1])) / tot;
    };
    history.lasso.push(gap(res.lasso));
    history.enet.push(gap(res.enet));
    if (Math.abs(res.lasso[0]) > Math.abs(res.lasso[1])) history.lassoWinA++;
    if (Math.abs(res.enet[0]) > Math.abs(res.enet[1])) history.enetWinA++;

    const pct = (v) => `${(v * 100).toFixed(0)}%`;
    const winner = (b) => (Math.abs(b[0]) > Math.abs(b[1]) ? 'A' : 'B');
    statsEl.innerHTML =
      `<div class="slider-label">sample ${run} &nbsp; twin A / twin B</div>` +
      `<div class="slider-label" style="color:var(--cosmos)">lasso: <span class="value">${res.lasso[0].toFixed(2)}</span> / <span class="value">${res.lasso[1].toFixed(2)}</span> &nbsp; favours <span class="value">${winner(res.lasso)}</span> by ${pct(gap(res.lasso))}</div>` +
      `<div class="slider-label" style="color:var(--aqua)">elasticnet: <span class="value">${res.enet[0].toFixed(2)}</span> / <span class="value">${res.enet[1].toFixed(2)}</span> &nbsp; split within ${pct(gap(res.enet))}</div>`;

    const N = history.lasso.length;
    if (N > 1) {
      tallyEl.innerHTML =
        `<div class="slider-label" style="margin-top:.6rem">over <span class="value">${N}</span> repetitions of the experiment, ` +
        `average imbalance <span class="value" style="color:var(--cosmos)">${pct(d3.mean(history.lasso))}</span> lasso vs ` +
        `<span class="value" style="color:var(--aqua)">${pct(d3.mean(history.enet))}</span> elasticnet</div>` +
        `<div class="slider-label">lasso crowned twin A <span class="value">${history.lassoWinA}</span> times and twin B ` +
        `<span class="value">${N - history.lassoWinA}</span> times. The winner is decided by noise.</div>`;
    }
  }

  document.querySelector('#collinear-resample').addEventListener('click', () => {
    run++;
    render(solve(drawSample(1000 + run * 7919, n, p - 2)), false);
  });

  // first view: the shipped sample, so the numbers match the scikit-learn reference
  const m0 = makeModel(col.X, col.y);
  const base = { lasso: m0.fit(ALPHA, 1), enet: m0.fit(ALPHA, L1_RATIO_EN) };
  const drift = d3.max(base.lasso.map((v, j) => Math.abs(v - col.ref.lasso[j])));
  if (drift > 0.05) console.warn(`collinear lasso mismatch vs sklearn: max |Δ| = ${drift.toFixed(3)}`);
  render(base, true);
}
