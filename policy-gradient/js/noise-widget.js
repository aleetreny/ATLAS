/* How wrong the estimate is, against how many episodes went into it.
 *
 * Two measures because they say different things: how far the estimate is from
 * the gradient, and whether it even points the same way.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

const VIEWS = [
  { key: 'error', label: 'how far from the gradient' },
  { key: 'cosine', label: 'whether it points the same way' },
];

export function initNoiseWidget(data) {
  const G = data.gradient;
  const readout = document.querySelector('#noise-readout');
  const controls = d3.select('#noise-controls');
  let view = 'error';
  const eps = [...new Set(G.rows.map((d) => d.episodes))];
  const pick = (kind) => G.rows.filter((d) => d.kind === kind);

  const chart = makeChart('#noise-chart', {
    width: 640, height: 350, margin: { top: 48, right: 124, bottom: 56, left: 70 },
  });
  const { g, w, h, margin } = chart;
  const plot = g.append('g');
  const key = g.append('g').attr('transform', `translate(${w + 8}, 6)`);
  const title = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -30)
    .style('font-size', '12px');
  const sub = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -12)
    .style('font-size', '11.5px');

  function render() {
    const x = d3.scaleLog().domain([eps[0] * 0.8, eps[eps.length - 1] * 1.25]).range([0, w]);
    const field = view;
    const all = G.rows.map((d) => d[field]);
    const y = view === 'error'
      ? d3.scaleLog().domain([Math.min(...all) * 0.7, Math.max(...all) * 1.4]).range([h, 0])
      : d3.scaleLinear().domain([0, 1]).range([h, 0]);
    const yt = view === 'error' ? [0.03, 0.1, 0.3, 1] : null;
    g.selectAll('g.axis').remove();
    g.selectAll('g.grid').remove();
    g.selectAll('text.axis-label').remove();
    drawGrid(g, x, y, w, h, { xValues: eps, yValues: yt, yTicks: 5 });
    drawAxes(g, x, y, w, h, { xValues: eps, yValues: yt, yTicks: 5, xFmt: d3.format(',d'),
      yFmt: view === 'error' ? d3.format('.2f') : d3.format('.1f') });
    axisLabels(g, w, h, { x: 'episodes averaged into one estimate',
      y: view === 'error' ? 'distance to the gradient' : 'cosine with the gradient',
      leftBudget: margin.left });
    plot.selectAll('*').remove();
    [['plain', 'var(--primary)'], ['baseline', 'var(--anchor)']].forEach(([kind, col]) => {
      const rows = pick(kind);
      plot.append('path').attr('fill', 'none').attr('stroke', col).attr('stroke-width', 2.6)
        .attr('d', d3.line().x((d) => x(d.episodes)).y((d) => y(d[field]))(rows));
      plot.selectAll(`circle.${kind}`).data(rows).join('circle').attr('class', kind)
        .attr('cx', (d) => x(d.episodes)).attr('cy', (d) => y(d[field])).attr('r', 4)
        .attr('fill', col);
    });
    if (view === 'error') {
      plot.append('line').attr('x1', 0).attr('x2', w)
        .attr('y1', y(G.exact_norm)).attr('y2', y(G.exact_norm))
        .attr('stroke', '#8e9aa6').attr('stroke-width', 1.6).attr('stroke-dasharray', '5 4');
      /* at the left the two curves are near the top, so the reference reads
         there; at the right end it sat on top of the plain curve */
      plot.append('text').attr('class', 'chart-note').attr('x', 2).attr('y', y(G.exact_norm) - 6)
        .style('font-size', '10.5px')
        .text(`the length of the gradient: ${G.exact_norm}`);
    }
    key.selectAll('*').remove();
    [['var(--primary)', ['plain']], ['var(--anchor)', ['minus the value', 'of the state']]]
      .forEach(([col, lines], i) => {
        key.append('line').attr('x1', 0).attr('x2', 14).attr('y1', i * 40 + 6)
          .attr('y2', i * 40 + 6).attr('stroke', col).attr('stroke-width', 2.6);
        lines.forEach((t, j) => {
          key.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', i * 40 + 20 + j * 11)
            .style('font-size', '10.5px').text(t);
        });
      });
    title.text(view === 'error'
      ? 'how far a sampled gradient lands from the real one'
      : 'whether a sampled gradient even points the right way');
    sub.text(`${G.rows[0].repeats} estimates per point, averaged`);

    const one = pick('plain').find((d) => d.episodes === 1);
    const oneB = pick('baseline').find((d) => d.episodes === 1);
    const big = pick('plain')[pick('plain').length - 1];
    const bigB = pick('baseline')[pick('baseline').length - 1];
    /* the fewest episodes at which the baseline is already as close as plain
       ever gets: it always exists, because the baseline wins at the last point */
    const match = pick('baseline').find((d) => d.error <= big.error);
    /* every other point: nine columns of seven digits do not fit the column
       the readout lives in, and the chart already has all nine */
    const shown = eps.filter((_, i) => i % 2 === 0);
    const cells = (kind, f) => shown
      .map((e) => `<td>${pick(kind).find((d) => d.episodes === e)[f]}</td>`).join('');
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>episodes</th>${shown.map((e) => `<th>${e}</th>`).join('')}</tr>
        <tr><td>distance, plain</td>${cells('plain', 'error')}</tr>
        <tr><td>distance, baseline</td>${cells('baseline', 'error')}</tr>
        <tr><td>cosine, plain</td>${cells('plain', 'cosine')}</tr>
        <tr><td>cosine, baseline</td>${cells('baseline', 'cosine')}</tr>
      </table>
      <div class="gen-note">
        One episode gives an estimate <span class="value">${one.error}</span> away
        from a gradient whose whole length is ${G.exact_norm}, pointing in a
        direction whose cosine with the truth is
        <span class="value">${one.cosine}</span>: at right angles, near enough.
        Subtracting what each state is worth, which cannot change the average
        because it does not depend on the action, takes that to ${oneB.error} and
        ${oneB.cosine}. At ${big.episodes} episodes it is ${big.error} and
        ${big.cosine} plain, ${bigB.error} and ${bigB.cosine} with the baseline,
        and the baseline is already that close at
        <span class="value">${match.episodes}</span> episodes:
        ${(big.episodes / match.episodes).toFixed(0)} times fewer for the same
        accuracy. That is what a baseline buys, stated as a number of episodes
        rather than as the word "variance".
      </div>`;
  }

  controls.selectAll('button').data(VIEWS).join('button')
    .attr('class', (d) => `atlas-btn${d.key === view ? '' : ' ghost'}`)
    .text((d) => d.label)
    .on('click', (_, d) => {
      view = d.key;
      controls.selectAll('button').attr('class', (q) => `atlas-btn${q.key === view ? '' : ' ghost'}`);
      render();
    });
  render();
  return { render };
}
