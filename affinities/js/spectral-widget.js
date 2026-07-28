/* Spectral clustering, all four steps of it, live.
 *
 * The reader picks a dataset, picks how the graph decides what "near" means,
 * and moves the one parameter that decision comes with. The page then builds
 * the graph, computes the bottom eigenvectors of its normalised Laplacian with
 * a Lanczos iteration, and runs k-means in that space, on every move.
 *
 * Three views because there are three things to see and they are not the same
 * thing: the graph that was built, the space the eigenvectors bent the data
 * into, and the answer. The eigenvalue panel stays under all three, because
 * the story of the whole section is that its first few numbers already say how
 * many pieces there are.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, annotate } from '../../assets/js/chart.js';
import { knnGraph, rbfGraph, spectral, components, kmeans, ari } from './graphkit.js';
import { drawPoints, drawEdges, legend, colourOf } from './draw.js';

const VIEWS = [['graph', 'the graph'], ['embed', 'the bent space'], ['clusters', 'the answer']];
const MAX_EDGES = 1400;

export function initSpectralWidget(data) {
  const cases = Object.keys(data.spectral.sets);
  const state = { name: cases[0], mode: 'knn', view: 'graph' };
  const knnGrid = data.spectral.knn_grid;
  const gammaGrid = data.spectral.gamma_grid;

  const main = makeChart('#spec-chart', {
    width: 640, height: 420, margin: { top: 34, right: 26, bottom: 46, left: 54 },
  });
  const x = d3.scaleLinear().range([0, main.w]);
  const y = d3.scaleLinear().range([main.h, 0]);
  const edgeG = main.g.append('g');
  const dotG = main.g.append('g');
  const noteG = main.g.append('g');
  const keyG = main.g.append('g');

  const eig = makeChart('#spec-eigen', {
    width: 640, height: 210, margin: { top: 34, right: 30, bottom: 48, left: 56 },
  });
  const ex = d3.scaleBand().range([0, eig.w]).padding(0.28);
  const ey = d3.scaleLinear().range([eig.h, 0]);
  const barG = eig.g.append('g');
  const gapG = eig.g.append('g');

  const slider = document.querySelector('#spec-param');
  const readout = document.querySelector('#spec-readout');

  const buttons = (wrap, items, get, set) => {
    const node = document.querySelector(wrap);
    items.forEach(([key, label]) => {
      const b = document.createElement('button');
      b.dataset.key = key;
      b.className = `atlas-btn${get() === key ? '' : ' ghost'}`;
      b.textContent = label;
      b.addEventListener('click', () => {
        set(key);
        node.querySelectorAll('button').forEach((o) => {
          o.className = `atlas-btn${o.dataset.key === get() ? '' : ' ghost'}`;
        });
      });
      node.appendChild(b);
    });
  };

  buttons('#spec-cases', cases.map((c) => [c, c]), () => state.name, (k) => {
    state.name = k;
    setCase();
    render();
  });
  buttons('#spec-modes', [['knn', 'k nearest neighbours'], ['rbf', 'a gaussian on every pair']],
    () => state.mode, (k) => {
      state.mode = k;
      setMode();
      render();
    });
  buttons('#spec-views', VIEWS, () => state.view, (k) => {
    state.view = k;
    render();
  });

  function grid() {
    return state.mode === 'knn' ? knnGrid : gammaGrid;
  }

  function setMode() {
    const g = grid();
    slider.min = 0;
    slider.max = g.length - 1;
    slider.step = 1;
    slider.value = state.mode === 'knn'
      ? Math.max(0, g.indexOf(data.spectral.knn))
      : Math.max(0, g.findIndex((v) => v >= 1));
    document.querySelector('#spec-param-name').childNodes[0].nodeValue =
      state.mode === 'knn' ? 'neighbours ' : 'gamma ';
    slider.setAttribute('aria-label', state.mode === 'knn'
      ? 'how many neighbours each point is joined to'
      : 'how fast similarity falls off with distance');
  }

  function setCase() {
    const pts = data.datasets[state.name].points;
    x.domain(d3.extent(pts, (p) => p[0])).nice();
    y.domain(d3.extent(pts, (p) => p[1])).nice();
  }

  function render() {
    const pts = data.datasets[state.name].points;
    const n = pts.length;
    const k = data.datasets[state.name].k;
    const truth = data.datasets[state.name].true_labels;
    const param = grid()[+slider.value];
    document.querySelector('#spec-param-label').textContent =
      state.mode === 'knn' ? param : param.toLocaleString('en-US');

    const W = state.mode === 'knn' ? knnGraph(pts, param) : rbfGraph(pts, param);
    const sp = spectral(W, n, k);
    const comp = components(W, n, state.mode === 'knn' ? 0 : 1e-9);
    const a = ari(truth, sp.labels);
    const zeros = sp.eigenvalues.filter((v) => v < 1e-8).length;

    /* ---- the picture ---- */
    edgeG.selectAll('line').remove();
    noteG.selectAll('*').remove();
    keyG.selectAll('*').remove();
    if (state.view === 'embed') {
      /* The rows of the eigenvector matrix, which is where k-means actually
         runs. With two columns they land on a circle by construction, because
         every row was scaled to length one; with three, the first two are
         drawn and the readout says so. */
      const rows = sp.embedding;
      const xs = rows.map((r) => r[0]);
      const ys = rows.map((r) => (r.length > 1 ? r[1] : 0));
      x.domain([-1.08, 1.08]);
      y.domain([-1.08, 1.08]);
      drawGrid(main.g, x, y, main.w, main.h, { xTicks: 5, yTicks: 5 });
      drawAxes(main.g, x, y, main.w, main.h, { xTicks: 5, yTicks: 5 });
      axisLabels(main.g, main.w, main.h, { x: 'first eigenvector', y: 'second eigenvector' });
      noteG.append('circle').attr('cx', x(0)).attr('cy', y(0))
        .attr('r', Math.abs(x(1) - x(0))).attr('fill', 'none')
        .attr('stroke', 'var(--squidink)').attr('stroke-width', 1)
        .attr('stroke-dasharray', '4 4').attr('opacity', 0.35);
      drawPoints(dotG, xs.map((v, i) => [v, ys[i]]), sp.labels, x, y, { r: 3.6 });
      annotate(noteG, 4, -8, k > 2
        ? `two of the ${k} eigenvectors, rows scaled to length one`
        : `both eigenvectors, rows scaled to length one`);
    } else {
      setCase();
      drawGrid(main.g, x, y, main.w, main.h, { xTicks: 5, yTicks: 5 });
      drawAxes(main.g, x, y, main.w, main.h, { xTicks: 5, yTicks: 5 });
      axisLabels(main.g, main.w, main.h, { x: '', y: '' });
      if (state.view === 'graph') {
        const items = [];
        let hi = 0;
        for (let i = 0; i < n; i++) {
          for (let j = i + 1; j < n; j++) {
            const w = W[i * n + j];
            if (w > 1e-4) {
              items.push({ a: i, b: j, w });
              if (w > hi) hi = w;
            }
          }
        }
        items.sort((p, q) => q.w - p.w);
        const shown = items.slice(0, MAX_EDGES);
        drawEdges(edgeG, shown.map((d) => ({ ...d, w: d.w / hi })), pts, x, y,
          { colour: 'var(--primary)' });
        drawPoints(dotG, pts, comp.count > 1 ? comp.comp : null, x, y, { r: 3 });
        annotate(noteG, 4, -8, items.length > shown.length
          ? `${items.length.toLocaleString('en-US')} edges, the ${MAX_EDGES.toLocaleString('en-US')} heaviest drawn`
          : `${items.length.toLocaleString('en-US')} edges, all drawn`);
        legend(keyG, [{
          colour: 'var(--squidink)',
          shape: 'dot',
          label: comp.count > 1 ? `coloured by connected piece (${comp.count})` : 'one connected piece',
        }], { x0: 4, y0: -24, gap: 200 });
      } else {
        drawPoints(dotG, pts, sp.labels, x, y, { r: 3.6 });
        annotate(noteG, 4, -8, `k-means with k = ${k}, run on the eigenvectors, drawn back here`);
      }
    }

    /* ---- the spectrum ---- */
    const vals = sp.eigenvalues.slice(0, Math.min(7, sp.eigenvalues.length));
    ex.domain(vals.map((_, i) => i + 1));
    ey.domain([0, Math.max(0.02, ...vals) * 1.25]);
    drawGrid(eig.g, ex, ey, eig.w, eig.h, { yTicks: 4 });
    drawAxes(eig.g, ex, ey, eig.w, eig.h, { yTicks: 4, yFmt: d3.format('.2f') });
    axisLabels(eig.g, eig.w, eig.h, { x: 'eigenvalue of the laplacian, smallest first', y: 'value' });
    barG.selectAll('rect').data(vals).join('rect')
      .attr('x', (d, i) => ex(i + 1))
      .attr('width', ex.bandwidth())
      .attr('y', (d) => ey(Math.max(d, 0)))
      .attr('height', (d) => eig.h - ey(Math.max(d, 0)))
      .attr('fill', (d, i) => (i < zeros ? 'var(--primary)' : 'var(--stone)'))
      .attr('stroke', 'var(--squidink)').attr('stroke-width', 0.8);
    /* the largest jump between consecutive eigenvalues, which is the only
       thing on this page that has an opinion about k */
    const gaps = vals.slice(1).map((v, i) => v - vals[i]);
    const gi = gaps.indexOf(Math.max(...gaps));
    gapG.selectAll('*').remove();
    gapG.append('line')
      .attr('x1', ex(gi + 2) - ex.step() * 0.14).attr('x2', ex(gi + 2) - ex.step() * 0.14)
      .attr('y1', 0).attr('y2', eig.h)
      .attr('stroke', 'var(--cosmos)').attr('stroke-width', 2).attr('stroke-dasharray', '5 4');
    /* The label sits on whichever side of the line has room: past the fifth
       bar it would otherwise run off the right edge of the canvas. */
    const gapX = ex(gi + 2) - ex.step() * 0.14;
    gapG.append('text').attr('class', 'chart-annotation')
      .attr('x', gapX + (gapX > eig.w * 0.6 ? -6 : 6)).attr('y', 12)
      .attr('text-anchor', gapX > eig.w * 0.6 ? 'end' : 'start')
      .style('font-size', '0.62rem')
      .text(`biggest gap: k = ${gi + 1}`);
    /* Inside the top margin rather than off the right edge, where a right
       margin wide enough to hold it would be wider than the label. */
    gapG.append('text').attr('class', 'chart-annotation')
      .attr('x', 0).attr('y', -12)
      .style('font-size', '0.62rem').style('fill', 'var(--primary)')
      .text(zeros === 1
        ? 'one eigenvalue at zero: the graph is in one piece'
        : `${zeros} eigenvalues at zero: the graph is in ${zeros} pieces`);

    /* ---- the readout ---- */
    const off = data.spectral.sets[state.name];
    const bestKnn = off.best_knn;
    readout.innerHTML =
      `<table class="aff-table">
        <tr><th></th><th>ARI against the truth</th><th>clusters asked for</th><th>graph</th></tr>
        <tr><td>spectral, here</td><td><span class="value">${a.toFixed(3)}</span></td>
            <td>${k}</td>
            <td>${state.mode === 'knn' ? `${param} neighbours` : `gamma ${param}`}, ${comp.count === 1
              ? 'one connected piece' : `${comp.count} connected pieces`}</td></tr>
        <tr><td>k-means on the same points</td><td><span class="value">${off.kmeans_ari.toFixed(3)}</span></td>
            <td>${k}</td><td>no graph at all</td></tr>
      </table>` +
      `<div class="slider-label" style="margin-top:.6rem;line-height:1.5">` +
      `${zeros >= k
        ? `The graph has already fallen apart into ${zeros} piece${zeros === 1 ? '' : 's'}, so the `
          + `eigenvectors are those pieces' indicators and nothing is being approximated. `
        : `The graph is in ${comp.count} piece${comp.count === 1 ? '' : 's'}, so the split is being `
          + `made inside it, by the ${k === 2 ? 'first eigenvector above zero' : `${k - 1} eigenvectors above zero`}. `}`
      + `${a >= off.kmeans_ari + 0.005
        ? `That beats k-means on the raw coordinates by ${(a - off.kmeans_ari).toFixed(3)}.`
        : a <= off.kmeans_ari - 0.005
          ? `That is ${(off.kmeans_ari - a).toFixed(3)} worse than k-means on the raw coordinates: `
            + `bending the space is not free, and this graph bent it wrong.`
          : `That is the same as k-means on the raw coordinates, to three decimals.`}`
      + ` Best over the whole neighbour sweep for this dataset: ${bestKnn.ari.toFixed(3)} at `
      + `${bestKnn.k} neighbours.`
      + `</div>`;
  }

  setCase();
  setMode();
  slider.addEventListener('input', render);
  render();
  return { render };
}
