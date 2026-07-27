/* The reachability plot, and the line across it.
 *
 * OPTICS does not return a clustering. It returns a walk through the data and,
 * for each step, the density at which that point became reachable. Draw those
 * as bars in walk order and the clusters are the valleys.
 *
 * A horizontal line across this plot is not a metaphor for DBSCAN's radius: it
 * is exactly DBSCAN's radius, and the labels it produces are scikit-learn's
 * cluster_optics_dbscan, computed here in the page. Which is what makes the
 * trap so damning, because the reader can try every line there is.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { optics, extractDbscan, ari } from './relaxkit.js';
import { drawPoints, colourOf } from './draw.js';

const CASES = [
  { key: 'blobs', label: 'three blobs', eps0: 0.95 },
  { key: 'rings', label: 'rings', eps0: 0.4 },
  { key: 'trap', label: 'the trap', eps0: 0.4 },
];

export function initOpticsWidget(data) {
  const cache = {};
  const state = { i: 0, mode: 'cut', xi: 2 };

  /* The bottom margin holds two things, not one: the axis, and below it a lane
     for xi's brackets. At the old 54 the brackets were drawn straight through
     the tick labels and read as a magenta line struck across "0 100 200 300". */
  const reach = makeChart('#optics-reach', {
    width: 640, height: 292, margin: { top: 30, right: 24, bottom: 96, left: 56 },
  });
  const BRACKET_LANE = 34;
  const scat = makeChart('#optics-scatter', {
    width: 640, height: 330, margin: { top: 30, right: 24, bottom: 44, left: 52 },
  });
  const barG = reach.g.append('g');
  const brackG = reach.g.append('g');
  const dotG = scat.g.append('g');
  const rTitle = reach.g.append('text').attr('class', 'chart-annotation')
    .attr('x', reach.w / 2).attr('y', -12).attr('text-anchor', 'middle').style('font-size', '0.66rem');
  const sTitle = scat.g.append('text').attr('class', 'chart-annotation')
    .attr('x', scat.w / 2).attr('y', -12).attr('text-anchor', 'middle').style('font-size', '0.66rem');
  let blade = null;

  const caseRow = document.querySelector('#optics-cases');
  const modeRow = document.querySelector('#optics-modes');
  const slider = document.querySelector('#optics-eps');
  const epsWrap = document.querySelector('#optics-eps-wrap');
  const xiWrap = document.querySelector('#optics-xi-wrap');
  const xiSlider = document.querySelector('#optics-xi');
  const readout = document.querySelector('#optics-readout');

  const caseBtns = CASES.map((c, i) => {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = c.label;
    b.addEventListener('click', () => {
      state.i = i;
      slider.value = grid().indexOf(c.eps0) >= 0 ? grid().indexOf(c.eps0) : Math.round(grid().length / 3);
      render();
    });
    caseRow.appendChild(b);
    return b;
  });
  const modeBtns = [['cut', 'one radius: the line'], ['xi', 'xi: the shape']].map(([key, label]) => {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = label;
    b.addEventListener('click', () => {
      state.mode = key;
      render();
    });
    modeRow.appendChild(b);
    return b;
  });

  const grid = () => data.optics[CASES[state.i].key].cut_grid;

  function run() {
    const key = CASES[state.i].key;
    if (!cache[key]) {
      const pts = data.datasets[key].points;
      cache[key] = { pts, ...optics(pts, data.meta.min_samples) };
    }
    return cache[key];
  }

  function render() {
    const c = CASES[state.i];
    const D = data.optics[c.key];
    const ds = data.datasets[c.key];
    const r = run();
    const eps = grid()[+slider.value];
    const xiRow = D.xis[+xiSlider.value];

    const labels = state.mode === 'cut'
      ? extractDbscan(r.order, r.reach, r.core, eps)
      : xiRow.labels;

    /* the curve: reachability in walk order, with the first point's infinity
       parked at the top of the canvas rather than silently dropped */
    const finite = r.reach.filter((v) => Number.isFinite(v));
    const top = d3.max(finite) * 1.08;
    const x = d3.scaleLinear().domain([0, r.order.length]).range([0, reach.w]);
    /* Square root, not linear. On the trap the tight clusters sit around 0.05
       and the jump to the diffuse one is 1.5, so a linear axis flattens the
       whole left half into a line one pixel tall and hides the structure this
       widget exists to show. A monotone transform leaves the meaning of the
       horizontal line exactly as it was: above is above. */
    const y = d3.scaleSqrt().domain([0, top]).range([reach.h, 0]);
    drawGrid(reach.g, x, y, reach.w, reach.h, { xTicks: 5, yTicks: 4 });
    drawAxes(reach.g, x, y, reach.w, reach.h, { xTicks: 5, yTicks: 4 });
    /* Only the y label comes from the helper: the x one has to clear the
       bracket lane, and axisLabels parks it at a fixed h + 42. */
    axisLabels(reach.g, reach.w, reach.h, { y: 'reachability, root scale' });
    let xLab = reach.g.select('text.axis-label.x');
    if (xLab.empty()) xLab = reach.g.append('text').attr('class', 'axis-label x');
    xLab.attr('x', reach.w / 2).attr('y', reach.h + BRACKET_LANE + 42)
      .attr('text-anchor', 'middle').text('points, in the order optics walked them');

    const bw = Math.max(1, reach.w / r.order.length - 0.4);
    barG.selectAll('rect').data(r.order).join('rect')
      .attr('x', (d, i) => x(i))
      .attr('width', bw)
      .attr('y', (d) => y(Math.min(Number.isFinite(r.reach[d]) ? r.reach[d] : top, top)))
      .attr('height', (d) => reach.h - y(Math.min(Number.isFinite(r.reach[d]) ? r.reach[d] : top, top)))
      .attr('fill', (d) => (labels[d] < 0 ? 'var(--stone)' : colourOf(labels[d])))
      .attr('opacity', 0.9);

    /* xi does not draw a line, it reads the steep sides of the valleys, and
       what it finds is a hierarchy: clusters inside clusters */
    const brackets = state.mode === 'xi' ? xiRow.hierarchy : [];
    /* Depth against EVERY other bracket, not just the earlier ones: the parent
       of a range comes after it in this list, so counting backwards put the
       whole hierarchy on one row and drew it as a single line. */
    const levels = brackets.map((b) => brackets.filter((o) => (o[1] - o[0]) > (b[1] - b[0])
      && o[0] <= b[0] && o[1] >= b[1]).length);
    brackG.selectAll('path').data(brackets).join('path')
      .attr('d', (b, i) => {
        const yy = reach.h + BRACKET_LANE + levels[i] * 9;
        return `M${x(b[0])},${yy - 5} L${x(b[0])},${yy} L${x(b[1] + 1)},${yy} L${x(b[1] + 1)},${yy - 5}`;
      })
      .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 1.8);

    if (state.mode === 'cut') {
      if (!blade) {
        blade = reach.g.append('g').attr('class', 'blade');
        blade.append('line').attr('x1', 0).attr('x2', reach.w)
          .attr('stroke', 'var(--cosmos)').attr('stroke-width', 2.4).attr('stroke-dasharray', '8 5');
        blade.append('rect').attr('x', 0).attr('width', reach.w).attr('y', -11).attr('height', 22)
          .attr('fill', 'transparent').style('cursor', 'ns-resize');
        blade.append('text').attr('class', 'chart-annotation').attr('x', reach.w - 4).attr('y', -8)
          .attr('text-anchor', 'end').style('font-size', '0.62rem').style('text-transform', 'none');
        blade.call(d3.drag().on('drag', (event) => {
          const v = y.invert(Math.max(0, Math.min(event.y, reach.h)));
          const g = grid();
          let bestI = 0;
          g.forEach((e, i) => { if (Math.abs(e - v) < Math.abs(g[bestI] - v)) bestI = i; });
          slider.value = bestI;
          render();
        }));
      }
      blade.style('display', null).attr('transform', `translate(0,${y(Math.min(eps, top))})`);
      blade.select('text').text(`eps ${eps}`);
    } else if (blade) {
      blade.style('display', 'none');
    }

    /* the same answer, on the points */
    const sx = d3.scaleLinear().domain(d3.extent(ds.points, (p) => p[0])).nice().range([0, scat.w]);
    const sy = d3.scaleLinear().domain(d3.extent(ds.points, (p) => p[1])).nice().range([scat.h, 0]);
    drawGrid(scat.g, sx, sy, scat.w, scat.h, { xTicks: 5, yTicks: 5 });
    drawAxes(scat.g, sx, sy, scat.w, scat.h, { xTicks: 5, yTicks: 5 });
    drawPoints(dotG, ds.points, labels, sx, sy, { r: 3.4 });

    const nClusters = Math.max(-1, ...labels) + 1;
    const nNoise = labels.filter((l) => l === -1).length;
    const a = ari(ds.true_labels, labels);
    caseBtns.forEach((b, i) => b.classList.toggle('ghost', i !== state.i));
    modeBtns.forEach((b, i) => b.classList.toggle('ghost', ['cut', 'xi'][i] !== state.mode));
    epsWrap.style.display = state.mode === 'cut' ? '' : 'none';
    xiWrap.style.display = state.mode === 'xi' ? '' : 'none';
    document.querySelector('#optics-eps-label').textContent = eps.toFixed(3);
    document.querySelector('#optics-xi-label').textContent = xiRow.xi;
    rTitle.text(state.mode === 'cut'
      ? 'a horizontal line here is a dbscan radius, exactly'
      : 'xi reads the steep sides, and finds clusters inside clusters');
    sTitle.text(`${nClusters} cluster${nClusters === 1 ? '' : 's'}, ${nNoise} noise point${nNoise === 1 ? '' : 's'}`);

    readout.innerHTML = verdict(c, D, labels, { eps, xi: xiRow.xi, nClusters, nNoise, a });
  }

  /* Every line below is a comparison against something measured, and on the
     trap the comparison is deliberately NOT the ARI: the DBSCAN article showed
     that calling the whole diffuse cluster noise scores 1.000 there. */
  function verdict(c, D, labels, s) {
    const head =
      `<table class="rel-table">
        <tr><th>${state.mode === 'cut' ? 'radius' : 'xi'}</th><th>clusters</th><th>noise</th><th>ARI</th></tr>
        <tr><td>${state.mode === 'cut' ? s.eps : s.xi}</td>
            <td><span class="value">${s.nClusters}</span></td>
            <td><span class="value">${s.nNoise}</span></td>
            <td><span class="value">${s.a.toFixed(3)}</span></td></tr>
      </table>`;
    let body;
    if (c.key === 'trap') {
      const m = trapMetrics(labels, data.datasets.trap.true_labels);
      const bestCut = D.cuts.filter((r) => r.separated)
        .reduce((a, b) => (b.recovered > a.recovered ? b : a));
      const bestXi = D.xis.reduce((a, b) => (Math.min(b.separated, b.recovered) > Math.min(a.separated, a.recovered) ? b : a));
      body = `The two tight clusters are ${m.sep ? 'separated' : '<span class="bold">fused</span>'}, `
        + `and <span class="bold">${(m.rec * 100).toFixed(0)}%</span> of the diffuse cluster is inside one cluster. `
        + `Both at once is what no single radius can do: of the ${D.cuts.length} lines in the sweep, the ones that `
        + `separate the pair keep at most ${(bestCut.recovered * 100).toFixed(0)}% of the diffuse cluster. `
        + `Xi at ${bestXi.xi} gets ${(bestXi.recovered * 100).toFixed(0)}%, against HDBSCAN's `
        + `${(D.hdbscan_recovered * 100).toFixed(0)}% in the previous article. `
        + `Watch the ARI while you do it: the line that scores highest, ${D.best_cut.ari.toFixed(3)} at eps `
        + `${D.best_cut.eps}, gets there by calling all 150 diffuse points noise.`;
    } else {
      const bestCut = D.best_cut;
      const bestXi = D.xis.reduce((a, b) => (b.ari > a.ari ? b : a));
      body = `Best line anywhere in the sweep: ARI ${bestCut.ari.toFixed(3)} at eps ${bestCut.eps}, with `
        + `${bestCut.n_clusters} clusters. Best xi: ${bestXi.ari.toFixed(3)} at ${bestXi.xi}. `
        + (bestCut.ari >= bestXi.ari
          ? `Here the line wins, and that is the honest result: when one radius does fit the data, `
            + `reading the shape instead buys nothing.`
          : `Here the shape wins, because the valleys sit at different depths.`);
    }
    return head + `<div class="slider-label" style="margin-top:.6rem;line-height:1.5">${body}</div>`;
  }

  function trapMetrics(labels, truth) {
    const major = (cls) => {
      const counts = new Map();
      labels.forEach((l, i) => {
        if (truth[i] === cls && l >= 0) counts.set(l, (counts.get(l) || 0) + 1);
      });
      if (!counts.size) return -9 - cls;
      return [...counts.entries()].reduce((a, b) => (b[1] > a[1] ? b : a))[0];
    };
    const counts = new Map();
    let total = 0;
    labels.forEach((l, i) => {
      if (truth[i] === 2) {
        total++;
        if (l >= 0) counts.set(l, (counts.get(l) || 0) + 1);
      }
    });
    const rec = counts.size ? Math.max(...counts.values()) / total : 0;
    return { sep: major(0) !== major(1), rec };
  }

  slider.min = 0;
  slider.max = grid().length - 1;
  slider.step = 1;
  slider.value = grid().indexOf(CASES[0].eps0);
  xiSlider.min = 0;
  xiSlider.max = data.optics.blobs.xis.length - 1;
  xiSlider.step = 1;
  xiSlider.value = state.xi;
  slider.addEventListener('input', render);
  xiSlider.addEventListener('input', render);
  render();
}
