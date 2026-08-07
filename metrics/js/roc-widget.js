/* The curve, then the two things it cannot see.
 *
 * The second view is the one that matters most to this atlas, because the
 * calibration article measured the same fact from the other side. Squashing the
 * scores towards a half is a monotone map: it changes no ordering, therefore it
 * changes no area, and it wrecks every probability the model reports. Drawing
 * the area flat while the Brier score moves is the cleanest way to say what
 * "the area is a statement about an ordering" means.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initRocWidget(data) {
  const buttons = document.querySelector('#roc-buttons');
  const readout = document.querySelector('#roc-readout');
  const st = { view: 'curve' };

  const chart = makeChart('#roc-chart', {
    width: 640, height: 380, margin: { top: 52, right: 40, bottom: 58, left: 80 },
  });
  const { g, w, h } = chart;
  const layer = g.append('g');

  const btns = {};
  for (const [key, text] of [['curve', 'the curve'],
    ['monotone', 'squashing the scores'], ['prevalence', 'changing the prevalence']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', () => { st.view = key; render(); });
    buttons.appendChild(b);
    btns[key] = b;
  }

  const y0 = data.roc.y;
  const p0 = data.roc.p;
  const order = p0.map((v, i) => i).sort((a, b) => p0[b] - p0[a]);
  const P = y0.reduce((s, v) => s + v, 0);
  const Nn = y0.length - P;
  const curve = [[0, 0]];
  let tp = 0;
  let fp = 0;
  order.forEach((i) => {
    if (y0[i] === 1) tp += 1; else fp += 1;
    curve.push([fp / Nn, tp / P]);
  });

  function render() {
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', k !== st.view));
    layer.selectAll('*').remove();

    if (st.view === 'curve') {
      const x = d3.scaleLinear().domain([0, 1]).range([0, w]);
      const y = d3.scaleLinear().domain([0, 1]).range([h, 0]);
      drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
      drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 5, xFmt: d3.format('.1f'), yFmt: d3.format('.1f') });
      axisLabels(g, w, h, { x: 'false positive rate', y: 'true positive rate' });
      layer.append('line').attr('x1', 0).attr('y1', h).attr('x2', w).attr('y2', 0)
        .attr('stroke', 'var(--squidink)').attr('stroke-width', 1).attr('stroke-dasharray', '4 4');
      layer.append('path').attr('d', d3.line().x((p) => x(p[0])).y((p) => y(p[1]))(curve))
        .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2.4);
      layer.append('text').attr('class', 'chart-note').attr('x', w * 0.42).attr('y', h * 0.62)
        .style('font-size', '11px').attr('fill', 'var(--squidink)')
        .text('a coin, which has area one half');
    } else if (st.view === 'monotone') {
      const rows = data.monotone.rows.filter((r) => r.k != null);
      const x = d3.scaleLog().domain([rows[0].k * 0.8, rows[rows.length - 1].k * 1.25])
        .range([0, w]);
      const y = d3.scaleLinear().domain([0, 1.02]).range([h, 0]);
      drawGrid(g, x, y, w, h, { xValues: rows.map((r) => r.k), yTicks: 5 });
      drawAxes(g, x, y, w, h, { xValues: rows.map((r) => r.k), yTicks: 5, xFmt: (v) => String(v) });
      axisLabels(g, w, h, { x: 'how hard the scores are squashed towards a half', y: '' });
      [['auc', 'var(--primary)', 'area under the curve'],
        ['brier', 'var(--cosmos)', 'brier score, lower is better'],
        ['ece', 'var(--anchor)', 'calibration error']].forEach(([key, colour, name], i) => {
        layer.append('path').attr('d', d3.line().x((r) => x(r.k)).y((r) => y(r[key]))(rows))
          .attr('fill', 'none').attr('stroke', colour).attr('stroke-width', 2.2);
        rows.forEach((r) => layer.append('circle').attr('cx', x(r.k)).attr('cy', y(r[key]))
          .attr('r', 3.6).attr('fill', colour));
        layer.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -36 + i * 20)
          .style('font-size', '11px').attr('fill', colour).text(name);
      });
    } else {
      const rows = data.prevalence.rows;
      const x = d3.scaleLog().domain([rows[rows.length - 1].prevalence * 0.7, rows[0].prevalence * 1.3])
        .range([0, w]);
      const y = d3.scaleLinear().domain([0, 1.02]).range([h, 0]);
      drawGrid(g, x, y, w, h, { xValues: rows.map((r) => r.prevalence), yTicks: 5 });
      drawAxes(g, x, y, w, h, {
        xValues: rows.map((r) => r.prevalence), yTicks: 5, xFmt: d3.format('.2f'),
      });
      axisLabels(g, w, h, { x: 'share of the rows that are positive', y: '' });
      [['auc', 'var(--primary)', 'area under the ROC curve', null],
        ['ap', 'var(--cosmos)', 'average precision', null],
        ['ap_chance', 'var(--squidink)', 'what a coin scores on average precision', '4 4']]
        .forEach(([key, colour, name, dash], i) => {
          layer.append('path').attr('d', d3.line().x((r) => x(r.prevalence)).y((r) => y(r[key]))(rows))
            .attr('fill', 'none').attr('stroke', colour).attr('stroke-width', 2.2)
            .attr('stroke-dasharray', dash);
          layer.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -36 + i * 20)
            .style('font-size', '11px').attr('fill', colour).text(name);
        });
    }

    const M = data.monotone;
    const PR = data.prevalence;
    const first = PR.rows[0];
    const last = PR.rows[PR.rows.length - 1];
    const base = M.base;
    readout.innerHTML = `<div class="gen-note">The two routes to the area agree to `
      + `<span class="bold">${base.auc_gap.toExponential(1)}</span>, which is float64 `
      + `arithmetic and not a tolerance anybody chose. Squashing the scores towards a half `
      + `moves the area by at most ${M.worst_shift.toExponential(1)} across the whole sweep, `
      + `and over the same sweep the Brier score runs from `
      + `${Math.min(...M.rows.map((r) => r.brier)).toFixed(4)} to `
      + `${Math.max(...M.rows.map((r) => r.brier)).toFixed(4)} and the calibration error `
      + `from ${Math.min(...M.rows.map((r) => r.ece)).toFixed(4)} to `
      + `${Math.max(...M.rows.map((r) => r.ece)).toFixed(4)}. That is the `
      + `<a href="../calibration/">calibration article</a>'s finding from the other side: a `
      + `perfectly ordered set of scores can be arbitrarily badly calibrated, and the area `
      + `cannot tell. Dropping the prevalence from ${(first.prevalence * 100).toFixed(0)}% to `
      + `${(last.prevalence * 100).toFixed(0)}% leaves the area at ${last.auc.toFixed(4)} `
      + `against ${first.auc.toFixed(4)}, moving ${Math.abs(last.auc - first.auc).toFixed(4)}, `
      + `while average precision falls from ${first.ap.toFixed(4)} to `
      + `${last.ap.toFixed(4)}. Neither is wrong: they answer different questions, and only `
      + `one of them changes when the base rate does.</div>`;
  }

  render();
  return { render, state: st };
}
