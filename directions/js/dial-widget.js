/* The dial: the same cloud, handed over.
 *
 * The reference the reader is scored against is swept at the DIAL'S OWN
 * resolution, half a degree, not at some finer grid the slider cannot reach.
 * A best-reachable number computed more finely than the control is both
 * unbeatable and unreachable, which is the worst of both: the reader is told
 * they fell short of something the widget never let them have.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { equalScales, unit, project, budgetBar, centred } from './cloud.js';

export function initDialWidget(data) {
  const C = data.cloud;
  const W = 700;
  const H = 500;
  /* Room for the budget bar at -42, its labels at -48, and a title above
     those: at a top margin of 62 the title's ascenders were cut off. */
  const margin = { top: 78, right: 26, bottom: 46, left: 58 };
  const { g, w, h } = makeChart('#dial-chart', { width: W, height: H, margin });

  const slider = document.querySelector('#dial-angle');
  const angleLabel = document.querySelector('#dial-angle-label');
  const readout = document.querySelector('#dial-readout');
  const snap = document.querySelector('#dial-snap');
  const scaleRow = document.querySelector('#dial-scale-buttons');

  const state = { deg: 0, scale: 'raw' };
  const btns = {};
  for (const [key, label] of [['raw', 'original units'], ['std', 'standardised']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn' + (key === 'raw' ? '' : ' ghost');
    b.textContent = label;
    b.addEventListener('click', () => {
      state.scale = key;
      render();
    });
    scaleRow.appendChild(b);
    btns[key] = b;
  }

  const stubG = g.append('g');
  const lineG = g.append('g');
  const dotG = g.append('g');
  const shadowG = g.append('g');
  const budget = budgetBar(g, w, { y: -42 });
  const mainLine = lineG.append('line')
    .attr('stroke', 'var(--primary)').attr('stroke-width', 2.6);
  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -62).attr('text-anchor', 'middle')
    /* Absolute px, not rem: inside a 700-unit viewBox shown 343px wide on a
       phone the scale is 0.49, so anything under 10.2px renders below the 5px
       floor the rest of the site keeps to. */
    .style('font-size', '11px').style('text-transform', 'none');

  function render() {
    const cur = C[state.scale];
    const P = centred(cur.points);
    const { x, y } = equalScales(P, w, h);
    const reach = Math.hypot(x.domain()[1] - x.domain()[0], y.domain()[1] - y.domain()[0]) / 2;
    const pr = project(P, state.deg);

    drawGrid(g, x, y, w, h, { xTicks: 6, yTicks: 6 });
    drawAxes(g, x, y, w, h, { xTicks: 6, yTicks: 6 });
    axisLabels(g, w, h, {
      x: state.scale === 'raw' ? 'flavanoids (centred)' : 'flavanoids (standardised)',
      y: state.scale === 'raw' ? 'colour absorbance' : 'absorbance (standardised)',
    });

    const [ux, uy] = unit(state.deg);
    mainLine
      .attr('x1', x(-ux * reach)).attr('y1', y(-uy * reach))
      .attr('x2', x(ux * reach)).attr('y2', y(uy * reach));

    stubG.selectAll('line').data(P).join('line')
      .attr('stroke', 'var(--primary)').attr('stroke-width', 1).attr('opacity', 0.38)
      .attr('x1', (d) => x(d[0])).attr('y1', (d) => y(d[1]))
      .attr('x2', (d, i) => x(pr.shadows[i][0])).attr('y2', (d, i) => y(pr.shadows[i][1]));

    dotG.selectAll('circle').data(P).join('circle')
      .attr('cx', (d) => x(d[0])).attr('cy', (d) => y(d[1]))
      .attr('r', 3.4).attr('fill', 'var(--squidink)').attr('opacity', 0.5)
      .attr('stroke', 'white').attr('stroke-width', 0.6);

    shadowG.selectAll('circle').data(pr.shadows).join('circle')
      .attr('cx', (d) => x(d[0])).attr('cy', (d) => y(d[1]))
      .attr('r', 3).attr('fill', 'var(--primary)')
      .attr('stroke', 'white').attr('stroke-width', 0.7);

    budget.update(pr.captured, cur.total);

    angleLabel.textContent = `${state.deg.toFixed(1)}°`;
    for (const [k, b] of Object.entries(btns)) b.classList.toggle('ghost', k !== state.scale);
    title.text(state.scale === 'raw'
      ? 'as recorded, only centred'
      : 'each measurement divided by its own standard deviation');

    /* The identity is the claim the section makes, so print the sum rather
       than asserting it: if it ever stops being the total, it shows. */
    const sum = pr.captured + pr.lost;
    const share = pr.captured / cur.total;
    const best = cur.dial_best_share;
    const behind = best - share;
    const gapDeg = Math.min(Math.abs(state.deg - cur.dial_best_deg),
      180 - Math.abs(state.deg - cur.dial_best_deg));

    let verdict;
    if (behind <= 1e-9) {
      verdict = `That is the best this dial can do. No other setting captures more, and the `
        + `direction it found is the first principal component: `
        + `${cur.exact_deg}° to the eigenvector's full precision.`;
    } else if (behind < 0.002) {
      verdict = `Within ${(behind * 100).toFixed(2)} points of the best reachable setting `
        + `(${cur.dial_best_share * 100}% at ${cur.dial_best_deg}°), and ${gapDeg.toFixed(1)}° `
        + `away from it. The peak is flat: that is why "roughly the long axis" is usually good `
        + `enough by eye and never good enough to publish.`;
    } else {
      verdict = `${(behind * 100).toFixed(2)} points short of the best reachable setting, `
        + `${cur.dial_best_share * 100}% at ${cur.dial_best_deg}°, which is ${gapDeg.toFixed(1)}° `
        + `from here. The worst you can do is ${cur.worst_share * 100}% at ${cur.worst_deg}°, `
        + `exactly perpendicular to the best.`;
    }

    readout.innerHTML = `
      <table class="dir-table">
        <tr><th>at ${state.deg.toFixed(1)}°</th><th>captured</th><th>thrown away</th><th>sum</th><th>share kept</th></tr>
        <tr>
          <td>${state.scale === 'raw' ? 'original units' : 'standardised'}</td>
          <td><span class="value">${pr.captured.toFixed(4)}</span></td>
          <td><span class="value">${pr.lost.toFixed(4)}</span></td>
          <td>${sum.toFixed(4)}</td>
          <td><span class="value">${(share * 100).toFixed(2)}%</span></td>
        </tr>
        <tr>
          <td>the whole cloud</td><td colspan="2">total variance</td>
          <td>${cur.total.toFixed(4)}</td><td>100%</td>
        </tr>
      </table>
      <div class="dir-note">${verdict}</div>`;
  }

  slider.addEventListener('input', () => {
    state.deg = +slider.value;
    render();
  });
  snap.addEventListener('click', () => {
    state.deg = C[state.scale].dial_best_deg;
    slider.value = String(state.deg);
    render();
  });

  render();
}
