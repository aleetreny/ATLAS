/* The leaderboard, with the two controls that decide whether any of it counts.
 *
 * Two views because there are two tasks and they do not agree. Ranking asks
 * whether the book the reader actually went on to rate five stars turns up in
 * a list of k; rating prediction asks for the number in the box. Recommending
 * the most rated book in the catalogue is a competitor at the first and cannot
 * play the second at all, which is exactly why it belongs on the chart.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, spread } from '../../assets/js/chart.js';

/* The labels live outside the plotting area on the right, so their length is a
   layout constraint and not a wording preference: the margin is 128 units and
   `.chart-note` at 11px runs about 6.5 units a character, so anything past
   eighteen characters leaves the canvas. */
const RANK_ROWS = [
  ['popular', 'most rated', 'var(--anchor)'],
  ['book_mean', 'best average', 'var(--smile)'],
  ['item_knn', 'neighbours, read', 'var(--primary)'],
  ['item_knn_rating', 'neighbours, stars', 'var(--cosmos)'],
  ['random', 'random', 'var(--stone)'],
];
const RMSE_ROWS = [
  ['global', 'one number for everyone'],
  ['book_mean', "the book's average"],
  ['user_mean', "the reader's average"],
  ['bias', 'reader plus book'],
  ['item_knn', 'item neighbours'],
];

export function initRankWidget(data) {
  const R = data.rank;
  const E = data.rating.rmse;
  const buttons = document.querySelector('#rank-buttons');
  const readout = document.querySelector('#rank-readout');
  const st = { view: 'rank' };

  const chart = makeChart('#rank-chart', {
    width: 640, height: 400, margin: { top: 46, right: 128, bottom: 58, left: 78 }, clip: true,
  });
  const { g, plot, w, h, clear } = chart;

  const btns = {};
  for (const [key, text] of [['rank', 'in a list of ten'], ['rmse', 'as a number in a box']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', () => { st.view = key; render(); });
    buttons.appendChild(b);
    btns[key] = b;
  }

  const ks = Object.keys(R.popular.hits).map(Number).sort((a, b) => a - b);
  const pct = (v, d = 1) => `${(100 * v).toFixed(d)}%`;

  function render() {
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', k !== st.view));
    clear();
    g.selectAll('text.chart-note').remove();

    if (st.view === 'rank') {
      const x = d3.scaleLog().domain([1, ks[ks.length - 1]]).range([0, w]);
      const maxY = d3.max(RANK_ROWS, ([key]) => d3.max(ks, (k) => R[key].hits[k]));
      const y = d3.scaleLinear().domain([0, maxY * 1.1]).range([h, 0]);
      drawGrid(g, x, y, w, h, { xValues: ks, yTicks: 5 });
      RANK_ROWS.forEach(([key, label, colour]) => {
        const pts = ks.map((k) => [k, R[key].hits[k]]);
        plot.append('path').attr('fill', 'none').attr('stroke', colour)
          .attr('stroke-width', key.startsWith('item') ? 2.4 : 1.8)
          .attr('stroke-dasharray', key === 'random' ? '4 3' : null)
          .attr('d', d3.line().x((p) => x(p[0])).y((p) => y(p[1]))(pts));
        plot.selectAll(null).data(pts).enter().append('circle')
          .attr('cx', (p) => x(p[0])).attr('cy', (p) => y(p[1])).attr('r', 2.6)
          .attr('fill', colour);
      });
      /* two of the five end within a couple of points of each other, so the
         labels are separated by value rather than by a hand written offset */
      const ends = spread(RANK_ROWS.map(([key, label, colour]) => ({
        label, colour, y: y(R[key].hits[ks[ks.length - 1]]) + 3.5, yMax: h + 10,
      })), 13);
      ends.forEach((d) => {
        g.append('text').attr('class', 'chart-note')
          .attr('x', w + 6).attr('y', d.y)
          .style('font-size', '11px').style('fill', d.colour)
          .text(d.label);
      });
      drawAxes(g, x, y, w, h, { xValues: ks, yTicks: 5, xFmt: (v) => String(v), yFmt: d3.format('.0%') });
      axisLabels(g, w, h, { x: 'length of the list', y: 'held out book is in it' });
      g.append('text').attr('class', 'chart-note')
        .attr('x', w / 2).attr('y', -26).attr('text-anchor', 'middle').style('font-size', '12px')
        .text(`${data.rank_meta.held_items.toLocaleString('en-US')} held out books that the reader gave four or five stars`);
    } else {
      const rows = RMSE_ROWS.filter(([key]) => E[key] !== undefined);
      const x = d3.scaleBand().domain(rows.map(([, l]) => l)).range([0, w]).padding(0.32);
      const lo = d3.min(rows, ([key]) => E[key]);
      const hi = d3.max(rows, ([key]) => E[key]);
      const y = d3.scaleLinear().domain([lo - (hi - lo) * 0.35, hi + (hi - lo) * 0.1]).range([h, 0]);
      drawGrid(g, x, y, w, h, { xValues: [], yTicks: 5 });
      plot.selectAll(null).data(rows).enter().append('rect')
        .attr('x', ([, l]) => x(l)).attr('width', x.bandwidth())
        .attr('y', ([key]) => y(E[key])).attr('height', ([key]) => h - y(E[key]))
        .attr('fill', ([key]) => (key === 'item_knn' ? 'var(--primary)'
          : key === 'bias' ? 'var(--anchor)' : 'var(--stone)'));
      rows.forEach(([key, label]) => {
        g.append('text').attr('class', 'chart-note')
          .attr('x', x(label) + x.bandwidth() / 2).attr('y', y(E[key]) - 7)
          .attr('text-anchor', 'middle').style('font-size', '11.5px')
          .text(E[key].toFixed(4));
        g.append('text').attr('class', 'chart-note')
          .attr('transform', `translate(${x(label) + x.bandwidth() / 2},${h + 14}) rotate(14)`)
          .attr('text-anchor', 'middle').style('font-size', '10.5px')
          .text(label);
      });
      drawAxes(g, x, y, w, h, { xValues: [], yTicks: 5, yFmt: d3.format('.2f') });
      axisLabels(g, w, h, { y: 'error on a held out rating' });
      g.append('text').attr('class', 'chart-note')
        .attr('x', w / 2).attr('y', -26).attr('text-anchor', 'middle').style('font-size', '12px')
        .text(`root mean squared error, ${data.meta.test_cells.toLocaleString('en-US')} held out ratings, lower is better`);
    }

    if (st.view === 'rank') {
      const pop = R.popular;
      const knn = R.item_knn;
      const ratio = knn.hits['10'] / pop.hits['10'];
      readout.innerHTML =
        `In a list of ten, item neighbours land the held out book `
        + `<span class="value">${pct(knn.hits['10'])}</span> of the time against `
        + `<span class="value">${pct(pop.hits['10'])}</span> for handing everybody the same ten most `
        + `rated books, a factor of <span class="value">${ratio.toFixed(2)}</span>. `
        + `Recommending the highest average rating instead scores `
        + `<span class="value">${pct(R.book_mean.hits['10'])}</span>, and random `
        + `<span class="value">${pct(R.random.hits['10'], 3)}</span>. `
        + `The difference that does not show up in the hit rate is the catalogue: the popular list `
        + `covers <span class="value">${pop.coverage}</span> distinct books across every reader, `
        + `and the neighbour list <span class="value">${knn.coverage.toLocaleString('en-US')}</span> of `
        + `${data.rank_meta.catalogue.toLocaleString('en-US')}.`;
    } else {
      readout.innerHTML =
        `Predicting the number is a different competition, and the neighbours barely win it: `
        + `<span class="value">${E.item_knn.toFixed(4)}</span> against `
        + `<span class="value">${E.bias.toFixed(4)}</span> for a model that is three numbers added `
        + `together and knows nothing about which books resemble which. Everything to the left of `
        + `those two is what you get without any personalisation at all: `
        + `<span class="value">${E.global.toFixed(4)}</span> for one number for everybody. `
        + `Nothing here can play the ranking game, because a rule for filling in a box says nothing `
        + `about which of ten thousand books to put in front of somebody.`;
    }
  }

  render();
  return { render, state: st };
}
