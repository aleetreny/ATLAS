/* The one decision the dataset made for itself.
 *
 * Three copies of the same sprites, differing only in how much per-pixel grain
 * sits on top of them, each with the same architecture trained on it for the
 * same number of steps from the same seed. The grainless arm is the generator
 * the rest of this page is about, not a re-run of it.
 *
 * The line is the shortcut, measured before any training: how often a single
 * number, how much of an image a three pixel blur takes away, separates that
 * arm's sprites from the same sprites drawn clean. The bars are what came out
 * the other end, scored by how saturated the thing the generator drew is,
 * against a dashed line at the real sprites' own value. The table carries the
 * rest, including the column that resolves the apparent contradiction between
 * the bars and the distance: an arm can be the most saturated of the three and
 * still be the worst, by drawing one shape over and over.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initGrainWidget(data) {
  const G = data.grain;
  if (!G) return;
  const rows = G.rows;
  const C = G.control;
  const label = (g) => (g === 0 ? 'no grain' : `grain ${g}`);

  const c = makeChart('#grain-chart', {
    width: 660, height: 360, margin: { top: 46, right: 132, bottom: 62, left: 78 },
  });
  const x = d3.scaleBand().domain(rows.map((r) => label(r.grain))).range([0, c.w]).padding(0.3);
  const top = Math.max(C.real.sat, ...rows.map((r) => r.sat)) * 1.28;
  const y = d3.scaleLinear().domain([0, top]).range([c.h, 0]);
  drawGrid(c.g, x, y, c.w, c.h, { yTicks: 5 });
  drawAxes(c.g, x, y, c.w, c.h, { yTicks: 5 });
  axisLabels(c.g, c.w, c.h, {
    x: 'what the sprites it was trained on were drawn with',
    y: 'colour of what it draws',
  });
  c.g.selectAll('rect.bar').data(rows).enter().append('rect').attr('class', 'bar')
    .attr('x', (r) => x(label(r.grain))).attr('y', (r) => y(r.sat))
    .attr('width', x.bandwidth()).attr('height', (r) => c.h - y(r.sat))
    .attr('fill', (r) => (r.grain === 0 ? 'var(--primary)' : 'var(--stone)'));
  /* Inside the bar rather than above it, because the shortcut line and its
     labels pass through exactly the band where a label above a bar would sit.
     The halo goes with it: `.chart-annotation` carries a white stroke so that
     text stays readable over a chart, and light text on a dark bar with a white
     halo fills its own counters and turns to mush, so the halo is repainted in
     whatever colour the label is standing on. */
  c.g.selectAll('text.v').data(rows).enter().append('text').attr('class', 'chart-annotation v')
    .attr('x', (r) => x(label(r.grain)) + x.bandwidth() / 2).attr('y', (r) => y(r.sat) + 20)
    .attr('text-anchor', 'middle').style('font-size', '11.5px').style('letter-spacing', '0.3px')
    .style('fill', (r) => (r.grain === 0 ? 'var(--paper)' : 'var(--squidink)'))
    .style('stroke', (r) => (r.grain === 0 ? 'var(--primary)' : 'var(--stone)'))
    .style('stroke-width', '2.5px')
    .text((r) => r.sat.toFixed(3));
  c.g.append('line').attr('x1', 0).attr('x2', c.w).attr('y1', y(C.real.sat)).attr('y2', y(C.real.sat))
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.4).attr('stroke-dasharray', '5 4');
  c.g.append('text').attr('class', 'chart-annotation').attr('x', c.w + 6)
    .attr('y', y(C.real.sat) + 4).style('font-size', '11.5px').style('letter-spacing', '0.4px')
    .style('fill', 'var(--squidink)').text('real sprites');

  /* the shortcut on its own scale: it is a probability, the bars are not */
  const y2 = d3.scaleLinear().domain([0.45, 1.06]).range([c.h, 0]);
  c.g.append('path').attr('fill', 'none').attr('stroke', 'var(--cosmos)').attr('stroke-width', 2.4)
    .attr('d', d3.line().x((r) => x(label(r.grain)) + x.bandwidth() / 2)
      .y((r) => y2(r.shortcut))(rows));
  c.g.selectAll('circle.s').data(rows).enter().append('circle').attr('class', 's')
    .attr('cx', (r) => x(label(r.grain)) + x.bandwidth() / 2).attr('cy', (r) => y2(r.shortcut))
    .attr('r', 4).attr('fill', 'var(--cosmos)');
  /* the halo, because the leftmost of these labels sits on top of a filled bar
     and a coloured text on a coloured bar is unreadable */
  c.g.selectAll('text.s').data(rows).enter().append('text').attr('class', 'chart-annotation s')
    .attr('x', (r) => x(label(r.grain)) + x.bandwidth() / 2).attr('y', (r) => y2(r.shortcut) - 10)
    .attr('text-anchor', 'middle').style('font-size', '11px').style('fill', 'var(--cosmos)')
    .style('paint-order', 'stroke').style('stroke', 'var(--paper)').style('stroke-width', '3px')
    .style('stroke-linejoin', 'round')
    .text((r) => `${(r.shortcut * 100).toFixed(0)}%`);
  c.g.append('text').attr('class', 'chart-annotation').attr('x', c.w + 6)
    .attr('y', y2(rows[rows.length - 1].shortcut) + 4)
    .style('font-size', '11.5px').style('letter-spacing', '0.4px').style('fill', 'var(--cosmos)')
    .text('the shortcut');

  const pc = (v) => `${(v * 100).toFixed(1)}%`;
  const cell = (v, fmt) => `<td>${v === undefined ? '' : fmt(v)}</td>`;
  /* Two tables rather than one of eight columns, because eight columns inside
     the article's own width is a table nobody reads without scrolling it. */
  const shortcutRow = (name, r) => `<tr><td>${name}</td>`
    + cell(r.shortcut, pc) + cell(r.shortcut_end, pc) + '</tr>';
  const outcomeRow = (name, r) => `<tr><td>${name}</td>`
    + cell(r.blobs, (v) => v.toFixed(2)) + cell(r.coverage, pc)
    + cell(r.sat, (v) => v.toFixed(3)) + cell(r.top_shape, pc)
    + cell(r.fd_blind, (v) => v.toFixed(0)) + '</tr>';
  const arms = (fn) => rows.map((r) => fn(label(r.grain), r)).join('');

  document.querySelector('#grain-readout').innerHTML =
    `<p>Three datasets of the same ${G.n.toLocaleString('en-US')} sprites, identical down to the `
    + `factors and differing only in the grain, one training run each at the page's own budget: `
    + `same architecture, same seed, ${G.steps.toLocaleString('en-US')} steps. The grainless arm `
    + `is not a re-run of the generator this page has been measuring, it is that generator, and `
    + `it is the first row of both tables.</p>`
    + `<p><span class="bold">The shortcut.</span> One number, how much of an image a three pixel `
    + `blur takes away, used as a classifier. Before is that arm's sprites against the same `
    + `sprites drawn clean, measured before any training exists; after is its real sprites against `
    + `what its generator ended up drawing.</p>`
    + `<table class="atlas-table"><thead><tr><th>Trained on</th>`
    + `<th>Before training</th><th>After training</th></tr></thead><tbody>`
    + arms(shortcutRow) + `</tbody></table>`
    + `<p><span class="bold">What came out.</span> The first four columns are what a sample `
    + `<span class="bold">is</span>, by the arithmetic the rest of this page uses, and the last is `
    + `a Frechet distance in the shape reader's features with the grain blurred off both sides. `
    + `The bottom two rows are the controls that give every column a top and a bottom.</p>`
    + `<table class="atlas-table"><thead><tr><th>Trained on</th>`
    + `<th>Patches</th><th>Area</th><th>Colour</th><th>Top shape</th>`
    + `<th>Distance</th></tr></thead><tbody>`
    + arms(outcomeRow)
    + outcomeRow('real sprites', { ...C.real, fd_blind: C.real_fd })
    + outcomeRow('flat grey', C.grey)
    + `</tbody></table>`;
}
