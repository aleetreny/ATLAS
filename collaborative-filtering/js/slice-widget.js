/* The average is hiding two different failures, so it is split two ways.
 *
 * By how much the reader has rated, and by how much the book has been rated.
 * The second view carries the number that decides whether any of this can be
 * deployed: the share of held out cells where the method has no opinion at
 * all, because the book has no neighbours the reader has touched.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initSliceWidget(data) {
  const S = data.slices;
  const buttons = document.querySelector('#slice-buttons');
  const readout = document.querySelector('#slice-readout');
  const st = { view: 'book' };

  const chart = makeChart('#slice-chart', {
    width: 640, height: 380, margin: { top: 46, right: 116, bottom: 62, left: 72 }, clip: true,
  });
  const { g, plot, w, h, clear } = chart;

  const btns = {};
  for (const [key, text] of [['book', 'by how known the book is'],
    ['user', 'by how much the reader has rated']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', () => { st.view = key; render(); });
    buttons.appendChild(b);
    btns[key] = b;
  }

  const pct = (v, d = 1) => `${(100 * v).toFixed(d)}%`;

  function render() {
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', k !== st.view));
    clear();
    g.selectAll('text.chart-note').remove();
    g.selectAll('g.axis.y2').remove();

    const rows = S[st.view];
    const x = d3.scalePoint().domain(rows.map((d) => d.decile)).range([0, w]).padding(0.5);
    const vals = rows.flatMap((d) => [d.knn, d.bias]);
    const y = d3.scaleLinear()
      .domain([d3.min(vals) - 0.02, d3.max(vals) + 0.02]).range([h, 0]);
    drawGrid(g, x, y, w, h, { xValues: [], yTicks: 5 });

    [['bias', 'reader plus book', 'var(--anchor)'],
      ['knn', 'item neighbours', 'var(--primary)']].forEach(([key, label, colour]) => {
      plot.append('path').attr('fill', 'none').attr('stroke', colour).attr('stroke-width', 2.2)
        .attr('d', d3.line().x((d) => x(d.decile)).y((d) => y(d[key]))(rows));
      plot.selectAll(null).data(rows).enter().append('circle')
        .attr('cx', (d) => x(d.decile)).attr('cy', (d) => y(d[key])).attr('r', 3)
        .attr('fill', colour);
      g.append('text').attr('class', 'chart-note')
        .attr('x', w + 6).attr('y', y(rows[rows.length - 1][key]) + 3.5)
        .style('font-size', '11px').style('fill', colour).text(label);
    });

    if (st.view === 'book') {
      const y2 = d3.scaleLinear().domain([0, 1]).range([h, 0]);
      plot.append('path').attr('fill', 'none').attr('stroke', 'var(--squidink)')
        .attr('stroke-width', 1.6).attr('stroke-dasharray', '5 3')
        .attr('d', d3.line().x((d) => x(d.decile)).y((d) => y2(d.covered))(rows));
      const ax2 = g.append('g').attr('class', 'axis y2')
        .attr('transform', `translate(${w},0)`)
        .call(d3.axisRight(y2).ticks(5).tickFormat(d3.format('.0%')));
      ax2.selectAll('text').style('fill', 'var(--squidink)');
      g.append('text').attr('class', 'chart-note')
        .attr('x', w + 6).attr('y', y2(rows[rows.length - 1].covered) + 14)
        .style('font-size', '11px').style('fill', 'var(--squidink)')
        .text('has an opinion');
    }

    drawAxes(g, x, y, w, h, { xValues: [], yTicks: 5, yFmt: d3.format('.2f') });
    axisLabels(g, w, h, { y: 'error on a held out rating' });
    rows.forEach((d) => {
      g.append('text').attr('class', 'chart-note')
        .attr('x', x(d.decile)).attr('y', h + 18).attr('text-anchor', 'middle')
        .style('font-size', '10.5px')
        .text(st.view === 'book' ? Math.round(d.popularity).toLocaleString('en-US')
          : Math.round(d.activity).toLocaleString('en-US'));
    });
    g.append('text').attr('class', 'chart-note')
      .attr('x', w / 2).attr('y', h + 40).attr('text-anchor', 'middle').style('font-size', '11.5px')
      .text(st.view === 'book' ? 'median ratings the book has, by decile'
        : 'median ratings the reader has given, by decile');
    g.append('text').attr('class', 'chart-note')
      .attr('x', w / 2).attr('y', -26).attr('text-anchor', 'middle').style('font-size', '12px')
      .text(st.view === 'book' ? 'the same error, split by how known the book is'
        : 'the same error, split by how much the reader has rated');

    const first = rows[0];
    const last = rows[rows.length - 1];
    readout.innerHTML = st.view === 'book'
      ? `In the least known tenth of the held out books (median `
        + `<span class="value">${Math.round(first.popularity)}</span> ratings each) the neighbours `
        + `have an opinion for <span class="value">${pct(first.covered)}</span> of the cells and `
        + `score <span class="value">${first.knn.toFixed(3)}</span> against the bias model's `
        + `<span class="value">${first.bias.toFixed(3)}</span>, which is `
        + `${first.knn > first.bias ? 'worse than doing nothing personal at all'
          : 'already better than doing nothing personal at all'}. In the best known tenth (median `
        + `<span class="value">${Math.round(last.popularity).toLocaleString('en-US')}</span>) they `
        + `have an opinion <span class="value">${pct(last.covered)}</span> of the time and score `
        + `<span class="value">${last.knn.toFixed(3)}</span> against `
        + `<span class="value">${last.bias.toFixed(3)}</span>. `
        + `${S.unreachable > 0
          ? `And ${S.unreachable.toLocaleString('en-US')} books in the catalogue are nobody's `
            + `neighbour: no list of ten can ever contain them.`
          : 'Every book in the catalogue is somebody\'s neighbour, so nothing is structurally unreachable.'}`
      : `From the least active tenth of readers (median `
        + `<span class="value">${Math.round(first.activity)}</span> ratings) to the most active `
        + `(median <span class="value">${Math.round(last.activity)}</span>), the neighbour error goes `
        + `from <span class="value">${first.knn.toFixed(3)}</span> to `
        + `<span class="value">${last.knn.toFixed(3)}</span> while the bias model goes from `
        + `<span class="value">${first.bias.toFixed(3)}</span> to `
        + `<span class="value">${last.bias.toFixed(3)}</span>. `
        + `${(first.knn - first.bias) * (last.knn - last.bias) < 0
          ? 'The two lines cross, which is the interesting case: personalisation is not worth the '
            + 'same at both ends of the site.'
          : 'The gap between them moves, but it never changes sign in this split.'}`;
  }

  render();
  return { render, state: st };
}
