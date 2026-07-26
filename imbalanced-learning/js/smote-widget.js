/* What resampling draws, on real habitat axes.
 *
 * The 4,700 background points are real training cells (700 positives, 4,000
 * negatives). "Photocopies" resamples positives with replacement and shows
 * copy counts as growing rings; "SMOTE" runs the actual algorithm live: for
 * each synthetic point, pick a random positive, pick one of its five nearest
 * positive neighbours, and land uniformly along the segment between them.
 * Both are seeded, so every visitor watches the same draw.
 */
import { drawGrid, drawAxes, axisLabels, makeChart } from '../../assets/js/chart.js';

function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const K = 5;

export function initSmoteWidget(data) {
  const pos = data.two_d.pos;
  const neg = data.two_d.neg;

  const { g, w, h } = makeChart('#sm-chart', {
    width: 700,
    height: 470,
    margin: { top: 34, right: 26, bottom: 56, left: 64 },
  });

  const x = d3.scaleLinear().domain(d3.extent([...pos, ...neg], (p) => p[0])).nice().range([0, w]);
  const y = d3.scaleLinear().domain(d3.extent([...pos, ...neg], (p) => p[1])).nice().range([h, 0]);
  drawGrid(g, x, y, w, h);
  drawAxes(g, x, y, w, h);
  axisLabels(g, w, h, { x: data.two_d.features[0], y: data.two_d.features[1] });

  g.append('g').selectAll('circle').data(neg).join('circle')
    .attr('cx', (p) => x(p[0])).attr('cy', (p) => y(p[1]))
    .attr('r', 2.2).attr('fill', 'var(--stone)').attr('opacity', 0.45);
  g.append('g').selectAll('circle').data(pos).join('circle')
    .attr('cx', (p) => x(p[0])).attr('cy', (p) => y(p[1]))
    .attr('r', 3).attr('fill', 'var(--primary)').attr('stroke', 'white')
    .attr('stroke-width', 0.6).attr('opacity', 0.85);

  const segG = g.append('g');
  const synthG = g.append('g');
  const ringG = g.append('g');
  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -14).attr('text-anchor', 'middle').style('font-size', '0.66rem');

  /* 5-NN among the positives, in the plotted units, standardised per axis so
   * neither metres-of-elevation nor metres-to-water dominates the metric.
   * 700 x 700 once, cached. */
  let nnCache = null;
  function neighbours() {
    if (nnCache) return nnCache;
    const sx = d3.deviation(pos, (p) => p[0]);
    const sy = d3.deviation(pos, (p) => p[1]);
    nnCache = pos.map((a, i) => {
      const ds = pos.map((b, j) => ({
        j,
        d: j === i ? Infinity : ((a[0] - b[0]) / sx) ** 2 + ((a[1] - b[1]) / sy) ** 2,
      }));
      ds.sort((u, v) => u.d - v.d);
      return ds.slice(0, K).map((e) => e.j);
    });
    return nnCache;
  }

  const readout = document.querySelector('#sm-readout');
  const slider = document.querySelector('#sm-slider');
  const btns = {
    none: document.querySelector('#sm-none'),
    copy: document.querySelector('#sm-copy'),
    smote: document.querySelector('#sm-smote'),
  };
  const state = { mode: 'none' };

  function render() {
    const n = +slider.value;
    document.querySelector('#sm-label').textContent = n;
    for (const [k, b] of Object.entries(btns)) b.classList.toggle('ghost', k !== state.mode);
    segG.selectAll('*').remove();
    synthG.selectAll('*').remove();
    ringG.selectAll('*').remove();

    if (state.mode === 'none') {
      title.text(`700 cottonwood cells (teal) among 4,000 others, real training data`);
      readout.innerHTML =
        `<div class="slider-label" style="line-height:1.5">The habitat band is real structure: low ` +
        `elevation, close to water. But notice how much grey lives inside the band too. The classes ` +
        `overlap, and no amount of resampling will change where the truth lives; it can only change ` +
        `how hard the model stares at each part of it.</div>`;
      return;
    }

    if (state.mode === 'copy') {
      const r = rng(41);
      const counts = new Map();
      for (let i = 0; i < n; i++) {
        const j = Math.floor(r() * pos.length);
        counts.set(j, (counts.get(j) || 0) + 1);
      }
      ringG.selectAll('circle').data([...counts.entries()]).join('circle')
        .attr('cx', ([j]) => x(pos[j][0])).attr('cy', ([j]) => y(pos[j][1]))
        .attr('r', ([, k2]) => 3 + 2.2 * Math.sqrt(k2))
        .attr('fill', 'none').attr('stroke', 'var(--secondary)').attr('stroke-width', 1.6)
        .attr('opacity', 0.8);
      const maxCopy = d3.max([...counts.values()]);
      title.text(`${n} photocopies spread over ${counts.size} of the 700 positives`);
      readout.innerHTML =
        `<div class="slider-label" style="line-height:1.5">Each ring is a point the sampler duplicated; ` +
        `ring size is its copy count (the most-copied point here appears ${maxCopy + 1} times in the new ` +
        `training set). Not one new location exists. The model sees the same evidence with some rows ` +
        `counted repeatedly, which is why oversampling and class weights land within a hair of each other ` +
        `in the benchmark below: weighting by 99 and photocopying 99 times are the same idea.</div>`;
      return;
    }

    /* SMOTE, live. */
    const nn = neighbours();
    const r = rng(43);
    const synth = [];
    const segs = [];
    for (let i = 0; i < n; i++) {
      const a = Math.floor(r() * pos.length);
      const b = nn[a][Math.floor(r() * K)];
      const t = r();
      synth.push([pos[a][0] + t * (pos[b][0] - pos[a][0]), pos[a][1] + t * (pos[b][1] - pos[a][1])]);
      if (segs.length < 60) segs.push([pos[a], pos[b]]);
    }
    segG.selectAll('line').data(segs).join('line')
      .attr('x1', (s) => x(s[0][0])).attr('y1', (s) => y(s[0][1]))
      .attr('x2', (s) => x(s[1][0])).attr('y2', (s) => y(s[1][1]))
      .attr('stroke', 'var(--secondary)').attr('stroke-width', 0.8).attr('opacity', 0.35);
    synthG.selectAll('circle').data(synth).join('circle')
      .attr('cx', (p) => x(p[0])).attr('cy', (p) => y(p[1]))
      .attr('r', 2.2).attr('fill', 'var(--cosmos)').attr('opacity', 0.75);
    title.text(`${n} synthetic positives (crimson), drawn between neighbour pairs`);
    readout.innerHTML =
      `<div class="slider-label" style="line-height:1.5">The first 60 parent segments are drawn faintly: ` +
      `every crimson point lies on a line between two real positives, never outside their neighbourhood ` +
      `graph. That is SMOTE's whole theory of the world: habitat is convex between neighbours. Mostly true ` +
      `here, and where the teal band mixes with grey, false as well: synthetic trees are being planted on ` +
      `cells whose real occupants are the majority class. SMOTE densifies the minority region; it cannot ` +
      `know where the minority region ends.</div>`;
  }

  for (const [k, b] of Object.entries(btns)) {
    b.addEventListener('click', () => {
      state.mode = k;
      render();
    });
  }
  slider.addEventListener('input', render);
  render();
}
