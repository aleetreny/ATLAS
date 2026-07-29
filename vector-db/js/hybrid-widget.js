/* Words against directions, and the two together.
 *
 * The scoring here is not recall against exhaustive search, because the
 * question has changed: with the newswire categories as an answer key, a run
 * can be scored on whether the documents it put at the top are about the same
 * thing as the query, discounted by position. That is the only measure on this
 * page that would change if the vectors got better rather than the index.
 *
 * The two query lengths are separate rows rather than an average because the
 * answer depends on them, and reporting one number would hide the finding.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

const ARMS = [
  ['sparse', 'word matching', 'var(--anchor)'],
  ['dense', 'vector similarity', 'var(--primary)'],
  ['fused', 'both, fused by rank', 'var(--cosmos)'],
  ['oracle', 'whichever of the two was better, per query', 'var(--smile)'],
];

export function initHybridWidget(data) {
  const H = data.hybrid;
  const readout = document.querySelector('#hybrid-readout');

  const chart = makeChart('#hybrid-chart', {
    width: 640, height: 350, margin: { top: 66, right: 34, bottom: 62, left: 84 },
  });
  const { g, w, h } = chart;
  const layer = g.append('g');

  const rows = H.rows;
  const groups = rows.map((r) => `${r.length} words`);
  const x = d3.scaleBand().domain(groups).range([0, w]).padding(0.3);
  const vals = rows.flatMap((r) => ARMS.map(([k]) => r[k]));
  const y = d3.scaleLinear().domain([0, Math.max(...vals) * 1.15]).range([h, 0]);
  drawGrid(g, x, y, w, h, { xValues: [], yTicks: 5 });
  drawAxes(g, x, y, w, h, { yTicks: 5, yFmt: d3.format('.2f') });
  axisLabels(g, w, h, { x: 'how many words the query keeps', y: 'discounted quality at ten' });

  const bw = x.bandwidth() / ARMS.length - 3;
  rows.forEach((r) => {
    ARMS.forEach(([key, , colour], i) => {
      const gx = x(`${r.length} words`) + i * (bw + 3);
      layer.append('rect').attr('x', gx).attr('width', bw)
        .attr('y', y(r[key])).attr('height', h - y(r[key]))
        .attr('fill', colour).attr('opacity', key === 'oracle' ? 0.45 : 1);
      layer.append('text').attr('class', 'chart-note').attr('x', gx + bw / 2)
        .attr('y', y(r[key]) - 6).attr('text-anchor', 'middle')
        .style('font-size', '10px').text(r[key].toFixed(3));
    });
  });

  ARMS.forEach(([, name, colour], i) => {
    layer.append('rect').attr('x', 0).attr('y', -56 + i * 14 - 8)
      .attr('width', 10).attr('height', 9).attr('fill', colour);
    layer.append('text').attr('class', 'chart-note').attr('x', 16).attr('y', -56 + i * 14)
      .style('font-size', '11px').attr('fill', colour).text(name);
  });

  const short = rows[0];
  const long = rows[rows.length - 1];
  const gain = (r) => r.fused - Math.max(r.sparse, r.dense);
  const tbl = rows.map((r) => `<tr><td>${r.length}</td><td>${r.queries}</td>`
    + `<td>${r.sparse.toFixed(4)}</td><td>${r.dense.toFixed(4)}</td>`
    + `<td>${r.fused.toFixed(4)}</td><td>${(r.dense_beats_sparse * 100).toFixed(0)}%</td>`
    + `<td>${(r.fused_beats_both * 100).toFixed(0)}%</td></tr>`).join('');
  readout.innerHTML = `<table class="gen-table"><thead><tr><th>query words</th>`
    + `<th>queries</th><th>word matching</th><th>vectors</th><th>fused</th>`
    + `<th>vectors won</th><th>fusion beat both</th></tr></thead><tbody>${tbl}</tbody></table>`
    + `<div class="gen-note">With ${short.length} words in the query, word matching scores `
    + `${short.sparse.toFixed(4)} and the vectors ${short.dense.toFixed(4)}; with `
    + `${long.length}, ${long.sparse.toFixed(4)} and ${long.dense.toFixed(4)}. `
    + (Math.sign(short.dense - short.sparse) !== Math.sign(long.dense - long.sparse)
      ? `<span class="bold">The winner changes with the query length</span>, which is why `
        + `these are two rows and not an average: a single number here would have been an `
        + `artefact of whichever length was chosen.`
      : `${long.dense > long.sparse ? 'The vectors win at both lengths' : 'Word matching wins at both lengths'}, `
        + `by ${Math.abs(long.dense - long.sparse).toFixed(4)} at the longer one.`)
    + ` Fusing them by rank alone, with no attempt to put two incomparable scores on one `
    + `scale, gives ${short.fused.toFixed(4)} and ${long.fused.toFixed(4)}, and beats both of `
    + `its inputs on ${(short.fused_beats_both * 100).toFixed(0)}% and `
    + `${(long.fused_beats_both * 100).toFixed(0)}% of individual queries. The faint bar is `
    + `the ceiling of choosing the better system per query, which nobody can do in advance: `
    + `fusion reaches ${((gain(long) / Math.max(1e-9, long.oracle - Math.max(long.sparse, long.dense))) * 100).toFixed(0)}% `
    + `of the way from the better single system to that ceiling at ${long.length} words. `
    + `The word matching side is BM25 over the same counts the `
    + `<a href="../tf-idf/">TF-IDF article</a> measured, and the queries are real documents `
    + `cut down to their commonest content words.</div>`;

  return { render: () => {} };
}
