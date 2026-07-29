/* The best a dot product can do, computed rather than trained.
 *
 * In the small world of this section the true score of every pair is known, so
 * the best rank k approximation of it is the truncated SVD and its error is a
 * number, not an outcome. The second view removes the cross term and the same
 * curve falls off a cliff at the true rank, which is the control: the ceiling
 * is not a property of dot products, it is a property of what is being
 * approximated.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initCeilWidget(data) {
  const C = data.ceiling;
  const buttons = document.querySelector('#ceil-buttons');
  const readout = document.querySelector('#ceil-readout');
  const st = { view: 'cross' };

  const chart = makeChart('#ceil-chart', {
    width: 640, height: 380, margin: { top: 46, right: 120, bottom: 58, left: 74 }, clip: true,
  });
  const { g, plot, w, h, clear } = chart;

  const btns = {};
  for (const [key, text] of [['cross', 'with a cross term'], ['pure', 'without one']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', () => { st.view = key; render(); });
    buttons.appendChild(b);
    btns[key] = b;
  }

  function render() {
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', k !== st.view));
    clear();
    g.selectAll('text.chart-note').remove();

    const rows = st.view === 'cross' ? C.rows : C.pure;
    const ks = rows.map((d) => d.k);
    const x = d3.scaleLog().domain([ks[0] * 0.85, ks[ks.length - 1] * 1.15]).range([0, w]);
    const lo = Math.max(d3.min(rows, (d) => d.rmse), 1e-8);
    const y = d3.scaleLog().domain([lo * 0.6, d3.max(rows, (d) => d.rmse) * 1.4]).range([h, 0]);
    const yTicks = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1].filter(
      (v) => v >= y.domain()[0] && v <= y.domain()[1]);
    drawGrid(g, x, y, w, h, { xValues: ks, yValues: yTicks });

    plot.append('path').attr('fill', 'none').attr('stroke', 'var(--primary)')
      .attr('stroke-width', 2.4)
      .attr('d', d3.line().x((d) => x(d.k)).y((d) => y(Math.max(d.rmse, lo * 0.61)))(rows));
    plot.selectAll(null).data(rows).enter().append('circle')
      .attr('cx', (d) => x(d.k)).attr('cy', (d) => y(Math.max(d.rmse, lo * 0.61))).attr('r', 3)
      .attr('fill', 'var(--primary)');
    g.append('text').attr('class', 'chart-note')
      .attr('x', w + 6).attr('y', y(Math.max(rows[rows.length - 1].rmse, lo * 0.61)) + 3.5)
      .style('font-size', '11px').style('fill', 'var(--primary)')
      .text('best possible');

    if (st.view === 'cross') {
      plot.append('line').attr('x1', x(C.d)).attr('x2', x(C.d)).attr('y1', 0).attr('y2', h)
        .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.2).attr('stroke-dasharray', '4 3');
      g.append('text').attr('class', 'chart-note')
        .attr('x', x(C.d) + 5).attr('y', 12).style('font-size', '11px')
        .text(`${C.d} tastes`);
    }

    drawAxes(g, x, y, w, h, { xValues: ks, yValues: yTicks, xFmt: (v) => String(v),
      yFmt: (v) => (v >= 0.01 ? d3.format('.2f')(v) : d3.format('.0e')(v)) });
    axisLabels(g, w, h, { x: 'numbers per side', y: 'best error any two tower model can reach' });
    g.append('text').attr('class', 'chart-note')
      .attr('x', w / 2).attr('y', -26).attr('text-anchor', 'middle').style('font-size', '12px')
      .text(`${C.n_users} readers by ${C.n_items} items, with the true score of every pair known`);

    const atD = rows.find((d) => d.k === C.d);
    const last = rows[rows.length - 1];
    readout.innerHTML = st.view === 'cross'
      ? `The truth here is ${C.d} tastes plus one cross term worth `
        + `<span class="value">${(100 * C.cross_share).toFixed(1)}%</span> of the total, and that `
        + `cross term makes the matrix full rank: <span class="value">${C.rank_needed}</span>. So a `
        + `two tower model with ${C.d} numbers a side cannot get below `
        + `<span class="value">${atD.rmse.toFixed(4)}</span> no matter how it is trained, and with `
        + `${last.k} it still cannot get below <span class="value">${last.rmse.toFixed(4)}</span>. `
        + `That is not an optimisation result, it is the singular values of the matrix it is trying `
        + `to write down.`
      : `Take the cross term out and the same curve falls off a cliff at exactly `
        + `<span class="value">${C.d}</span>, which is the number of tastes the world was built `
        + `with: <span class="value">${rows.find((d) => d.k === C.d).rmse.toExponential(2)}</span>, `
        + `machine noise. This is the control that makes the other view mean something. A dot `
        + `product is not a weak model; it is an exact model of a world with no interaction the two `
        + `sides cannot each describe on their own.`;
  }

  render();
  return { render, state: st };
}
