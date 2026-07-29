/* Which of the differences on the chart above are differences.
 *
 * The floor every article in this module draws answers one question: is a pool
 * distinguishable from real data. It does not answer the question this article
 * actually asks, which is whether one pool is closer to real data than another,
 * and after the paths are straightened the three fast arms land within one
 * percent of each other, which is exactly where a floor gets mistaken for a
 * resolution.
 *
 * So each cell was measured six times with nothing changed but the noise, and a
 * pair is called ranked only when six redraws of one never reach six redraws of
 * the other. The band comes from the two cells being compared and never from
 * the whole chart: the collapsed arm's value is thirty times the others', its
 * spread is proportionally larger, and one global band taken from it would have
 * declared every interesting comparison unresolvable.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';

export function initSpreadWidget(data) {
  const SP = data.spread;
  const buttons = document.querySelector('#spread-buttons');
  const readout = document.querySelector('#spread-readout');
  const st = { view: 'all' };

  /* the same spelling the pair table below uses, and short enough to fit the
     left margin: "rectified flow, 100 evals" did not, and ran off the edge */
  const label = (r) => (r.arm === 'distilled student'
    ? 'distilled student'
    : `${r.arm} x${r.steps}`);
  const key = (r) => `${r.arm} x${r.steps}`;

  const chart = makeChart('#spread-chart', {
    width: 640, height: 300, margin: { top: 48, right: 34, bottom: 60, left: 158 },
  });
  const { g, w, h } = chart;
  const layer = g.append('g');
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -26).attr('text-anchor', 'middle')
    .style('font-size', '12px');

  const btns = {};
  for (const [k, text] of [['all', 'all five'], ['fast', 'without the collapsed arm']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', () => { st.view = k; render(); });
    buttons.appendChild(b);
    btns[k] = b;
  }

  function render() {
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', k !== st.view));
    layer.selectAll('*').remove();
    g.selectAll('g.axis').remove();
    g.selectAll('text.axis-label').remove();

    const rows = st.view === 'all'
      ? SP.rows
      : SP.rows.filter((r) => Math.abs(r.mean) < 10 * Math.min(...SP.rows.map(
        (q) => Math.abs(q.mean))));
    const lows = rows.map((r) => Math.abs(r.lo));
    const highs = rows.map((r) => Math.abs(r.hi));
    const x = d3.scaleLog()
      .domain([Math.min(...lows) * 0.94, Math.max(...highs) * 1.06]).range([0, w]);
    const y = d3.scaleBand().domain(rows.map(key)).range([0, h]).padding(0.42);

    /* Five ticks spread evenly across whatever this view spans, rounded to two
       figures. Neither of the obvious alternatives works for both views: d3's
       own log ticks left the four arms that matter crowded against an
       unlabelled left edge, and a fixed list of decades put a single tick on
       the second view, whose whole domain is half a decade wide. */
    const [d0, d1] = x.domain();
    const dec = [];
    for (let i = 0; i < 5; i++) {
      const r = Number((d0 * (d1 / d0) ** (i / 4)).toPrecision(2));
      if (r >= d0 && r <= d1 && !dec.includes(r)) dec.push(r);
    }
    /* a band scale contributes no grid rules, which is what a categorical axis
       should do, so only the value axis is ruled */
    drawGrid(g, x, y, w, h, { xValues: dec });
    g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(x).tickValues(dec).tickFormat(d3.format('.1e')));
    g.append('g').attr('class', 'axis y')
      .call(d3.axisLeft(y).tickFormat((v) => label(rows.find((r) => key(r) === v))));
    axisLabels(g, w, h, { x: 'distance from real digits' });

    rows.forEach((r) => {
      const cy = y(key(r)) + y.bandwidth() / 2;
      const colour = r.arm === 'rectified flow' ? 'var(--squidink)'
        : (r.arm === 'after reflow' ? 'var(--primary)' : 'var(--secondary)');
      layer.append('line').attr('x1', x(Math.abs(r.lo))).attr('x2', x(Math.abs(r.hi)))
        .attr('y1', cy).attr('y2', cy)
        .attr('stroke', colour).attr('stroke-width', 9).attr('opacity', 0.28)
        .attr('stroke-linecap', 'round');
      [r.lo, r.hi].forEach((v) => {
        layer.append('line').attr('x1', x(Math.abs(v))).attr('x2', x(Math.abs(v)))
          .attr('y1', cy - 7).attr('y2', cy + 7)
          .attr('stroke', colour).attr('stroke-width', 1.4);
      });
      layer.append('circle').attr('cx', x(Math.abs(r.mean))).attr('cy', cy).attr('r', 4.6)
        .attr('fill', colour);
    });

    wrapLabel(title, st.view === 'all'
      ? 'each arm measured six times with nothing changed but the noise'
      : 'the same picture without the collapsed arm, which is the only way to see the overlap',
    w - 8);

    const ranked = SP.pairs.filter((p) => p.resolved);
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>this against that</th><th>apart by</th><th>in combined sd</th><th>rankable</th></tr>
        ${SP.pairs.map((p) => `<tr>
          <td>${p.a} <span style="opacity:0.6">vs</span> ${p.b}</td>
          <td><span class="value">${p.diff}</span></td>
          <td>${p.sigmas.toFixed(2)}</td>
          <td>${p.resolved ? '<span class="bold">yes</span>' : 'no'}</td>
        </tr>`).join('')}
      </table>
      <div class="gen-note">
        ${ranked.length} of these ${SP.pairs.length} comparisons survive, and
        ${ranked.every((p) => p.a.startsWith('rectified flow x1') || p.b.startsWith('rectified flow x1'))
    ? `every one of them involves the same cell: the plain rectified flow given a single
       evaluation. Take that one arm out and nothing on this page is rankable at all. That is
       not a failure of the experiment, it is the result: after the paths are straightened,
       spending a hundred evaluations instead of one buys a change smaller than the change you
       get from re-rolling the noise.`
    : `the rest are inside the spread the statistic shows when nothing changes but the noise.`}
        Every row is ${SP.draws} redraws of 250 samples each, and the "combined sd" column is the
        gap between two arms divided by the two arms' own standard deviations added in
        quadrature, so a row under about two is a row nobody should be ranking.
      </div>`;
  }

  render();
  return { render };
}
