/* The centrepiece: the averaging equation against what was measured.
 *
 * Log axes on both sides, because the whole point is the shape: a 1/B line
 * bending over onto a floor. The predicted curve is drawn dashed on top of the
 * observed points, and the floor gets its own rule, because "the term that does
 * not go away" is a horizontal line and should look like one.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, tooltip } from '../../assets/js/chart.js';

const SERIES = {
  bagging: { key: 'bagging', label: 'bagging', colour: 'var(--aqua)', btn: '#var-bag' },
  random_forest: { key: 'random_forest', label: 'random forest', colour: 'var(--primary)', btn: '#var-rf' },
  extra_trees: { key: 'extra_trees', label: 'extra trees', colour: 'var(--smile)', btn: '#var-et' },
};

export function initVarWidget(data) {
  const V = data.variance;
  const { g, w, h } = makeChart('#var-chart', {
    width: 700,
    height: 440,
    /* The y ticks are in scientific notation and run to six characters, so the
     * rotated axis label needs real room to its left. */
    margin: { top: 34, right: 130, bottom: 60, left: 100 },
  });

  const allCurves = Object.values(V).flatMap((v) => v.curve);
  const x = d3.scaleLog().domain([0.9, 240]).range([0, w]);
  const y = d3
    .scaleLog()
    .domain([
      d3.min(allCurves, (c) => Math.min(c.observed, c.predicted)) * 0.7,
      d3.max(allCurves, (c) => Math.max(c.observed, c.predicted)) * 1.4,
    ])
    .range([h, 0]);

  drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
  drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 5, yFmt: d3.format('.1e') });
  /* Tree counts, printed as tree counts rather than as SI prefixes. */
  g.select('g.axis.x').call(d3.axisBottom(x).tickValues([1, 3, 10, 30, 100, 200]).tickFormat(d3.format('d')).tickSizeOuter(0));
  axisLabels(g, w, h, {
    x: 'trees in the ensemble, B',
    y: 'variance of the prediction',
  });

  const floorG = g.append('g');
  const predG = g.append('g');
  const obsG = g.append('g');
  const dotG = g.append('g');
  const labG = g.append('g');
  const tip = tooltip();
  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -14).attr('text-anchor', 'middle').style('font-size', '0.66rem');

  const line = d3.line().x((c) => x(c.b));
  const readout = document.querySelector('#var-readout');
  const state = { key: 'bagging', all: false };

  function render() {
    const keys = state.all ? Object.keys(SERIES) : [state.key];

    floorG.selectAll('line').data(keys, (k) => k).join('line')
      .attr('x1', 0).attr('x2', w)
      .attr('y1', (k) => y(Math.max(V[k].floor, y.domain()[0])))
      .attr('y2', (k) => y(Math.max(V[k].floor, y.domain()[0])))
      .attr('stroke', (k) => SERIES[k].colour)
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '3 4')
      .attr('opacity', 0.7);

    predG.selectAll('path').data(keys, (k) => k).join('path')
      .attr('fill', 'none')
      .attr('stroke', (k) => SERIES[k].colour)
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '7 5')
      .attr('opacity', 0.9)
      .attr('d', (k) => line.y((c) => y(c.predicted))(V[k].curve));

    obsG.selectAll('path').data(keys, (k) => k).join('path')
      .attr('fill', 'none')
      .attr('stroke', (k) => SERIES[k].colour)
      .attr('stroke-width', 3)
      .attr('opacity', 0.35)
      .attr('d', (k) => line.y((c) => y(c.observed))(V[k].curve));

    const pts = keys.flatMap((k) => V[k].curve.map((c) => ({ ...c, k })));
    dotG.selectAll('circle').data(pts, (d) => `${d.k}-${d.b}`).join('circle')
      .attr('cx', (d) => x(d.b))
      .attr('cy', (d) => y(d.observed))
      .attr('r', 4.4)
      .attr('fill', (d) => SERIES[d.k].colour)
      .attr('stroke', 'white')
      .attr('stroke-width', 1.4)
      .on('mousemove', (event, d) => tip.show(
        `<span style="color:${SERIES[d.k].colour}">${SERIES[d.k].label}</span><br/>` +
        `B = ${d.b}<br/>measured ${d.observed.toExponential(2)}<br/>predicted ${d.predicted.toExponential(2)}`,
        event
      ))
      .on('mouseleave', () => tip.hide());

    labG.selectAll('text').data(keys, (k) => k).join('text')
      .attr('class', 'chart-annotation')
      .attr('x', w + 8)
      .attr('y', (k) => y(V[k].curve[V[k].curve.length - 1].observed) + 4)
      .style('font-size', '0.62rem').style('text-transform', 'none')
      .attr('fill', (k) => SERIES[k].colour)
      .text((k) => SERIES[k].label);

    title.text('solid: measured over 20 independent training sets.  dashed: the equation.');
    for (const s of Object.values(SERIES)) {
      document.querySelector(s.btn).classList.toggle('ghost', state.all || s.key !== state.key);
    }

    const shown = state.all ? Object.keys(SERIES) : [state.key];
    readout.innerHTML =
      `<table class="tr-table">
         <tr><th></th>${shown.map((k) => `<th style="color:${SERIES[k].colour}">${SERIES[k].label}</th>`).join('')}</tr>
         <tr><td>ρ, correlation between two trees</td>${shown.map((k) => `<td>${V[k].rho.toFixed(4)}</td>`).join('')}</tr>
         <tr><td>σ², variance of one tree</td>${shown.map((k) => `<td>${V[k].sigma2.toFixed(4)}</td>`).join('')}</tr>
         <tr><td>floor, ρσ²</td>${shown.map((k) => `<td>${V[k].floor.toExponential(2)}</td>`).join('')}</tr>
         <tr><td>accuracy, 1 tree → 200</td>${shown.map((k) => `<td>${V[k].single_acc.toFixed(4)} → ${V[k].ens_acc.toFixed(4)}</td>`).join('')}</tr>
       </table>` +
      `<div class="slider-label" style="margin-top:.6rem;line-height:1.5">${
        state.all
          ? `Reading down the ρ row is the argument of this section: sampling columns cuts the correlation ` +
            `between trees by <span class="value">${(V.bagging.rho / V.random_forest.rho).toFixed(1)}×</span>, ` +
            `which is the only reason a random forest beats plain bagging. Reading down σ² is the price: each ` +
            `individual tree gets worse as the handicap grows.`
          : `With no free parameters the equation tracks the measurement to within ` +
            `<span class="value">${(V[state.key].worst_rel_error * 100).toFixed(0)}%</span> everywhere on this ` +
            `curve. The dashed line flattens onto ρσ² and stays there: past a few dozen trees, adding more ` +
            `stops buying anything, and the only way further down is to make the trees disagree more.`
      }</div>`;
  }

  for (const s of Object.values(SERIES)) {
    document.querySelector(s.btn).addEventListener('click', () => {
      state.key = s.key;
      state.all = false;
      document.querySelector('#var-all').checked = false;
      render();
    });
  }
  document.querySelector('#var-all').addEventListener('change', (e) => {
    state.all = e.target.checked;
    render();
  });
  render();
}
