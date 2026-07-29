/* Per pattern, on a graph whose patterns were written on purpose.
 *
 * An average over eleven relations hides exactly the thing this article is
 * about, because the relations differ in kind and not only in difficulty. Here
 * each relation is one pattern: a symmetric one, a pair of inverses, a
 * composition, and a hierarchy where many entities share one parent. The bars
 * are the mean reciprocal rank on each, which makes the algebra visible as
 * three different shapes rather than three numbers.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

const MODELS = [['transe', 'TransE', 'var(--anchor)'], ['rotate', 'RotatE', 'var(--primary)'],
  ['distmult', 'DistMult', 'var(--cosmos)']];
const NICE = {
  mirror: 'symmetric', parent: 'hierarchy', child: 'its inverse',
  grandparent: 'composition', kind_of: 'many to one',
};

export function initPatWidget(data) {
  const P = data.patterns.models;
  const readout = document.querySelector('#pat-readout');

  const chart = makeChart('#pat-chart', {
    width: 640, height: 380, margin: { top: 44, right: 26, bottom: 66, left: 76 }, clip: true,
  });
  const { g, plot, w, h } = chart;

  const patterns = Object.keys(P.transe.per_pattern);
  const x0 = d3.scaleBand().domain(patterns).range([0, w]).padding(0.22);
  const x1 = d3.scaleBand().domain(MODELS.map((m) => m[0])).range([0, x0.bandwidth()]).padding(0.1);
  const y = d3.scaleLinear().domain([0, 1.05]).range([h, 0]);
  drawGrid(g, x0, y, w, h, { xValues: [], yTicks: 5 });

  patterns.forEach((pat) => {
    MODELS.forEach(([key, label, colour]) => {
      const v = P[key].per_pattern[pat].mrr;
      plot.append('rect')
        .attr('x', x0(pat) + x1(key)).attr('width', x1.bandwidth())
        .attr('y', y(v)).attr('height', h - y(v)).attr('fill', colour);
      g.append('text').attr('class', 'chart-note')
        .attr('x', x0(pat) + x1(key) + x1.bandwidth() / 2).attr('y', y(v) - 6)
        .attr('text-anchor', 'middle').style('font-size', '10px').text(v.toFixed(2));
    });
    g.append('text').attr('class', 'chart-note')
      .attr('x', x0(pat) + x0.bandwidth() / 2).attr('y', h + 18)
      .attr('text-anchor', 'middle').style('font-size', '11px')
      .text(NICE[pat] || pat);
  });

  MODELS.forEach(([key, label, colour], i) => {
    g.append('rect').attr('x', 6 + i * 108).attr('y', -34).attr('width', 10).attr('height', 10)
      .attr('fill', colour);
    g.append('text').attr('class', 'chart-note')
      .attr('x', 20 + i * 108).attr('y', -25).style('font-size', '11px').text(label);
  });

  drawAxes(g, x0, y, w, h, { xValues: [], yTicks: 5, yFmt: d3.format('.1f') });
  axisLabels(g, w, h, { y: 'mean reciprocal rank on that pattern' });
  g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', h + 40).attr('text-anchor', 'middle').style('font-size', '11.5px')
    .text(`${data.patterns.meta.triples.toLocaleString('en-US')} triples over `
      + `${data.patterns.meta.entities.toLocaleString('en-US')} entities, one relation per pattern`);

  const sym = { t: P.transe.per_pattern.mirror.mrr, r: P.rotate.per_pattern.mirror.mrr,
    d: P.distmult.per_pattern.mirror.mrr };
  const hier = { t: P.transe.per_pattern.parent.mrr, r: P.rotate.per_pattern.parent.mrr,
    d: P.distmult.per_pattern.parent.mrr };
  const comp = { t: P.transe.per_pattern.grandparent.mrr, r: P.rotate.per_pattern.grandparent.mrr,
    d: P.distmult.per_pattern.grandparent.mrr };
  /* Every sentence here is a comparison, so every sentence branches on its own
     comparison. The three claims are independent and the graph is written by
     hand, but "written by hand" is not a measurement. */
  const dSym = sym.d >= Math.min(sym.t, sym.r);
  const dDrops = hier.d < sym.d;
  const compFree = Math.min(comp.t, comp.r) > comp.d;
  readout.innerHTML =
    `Read it column by column, because each column is a different claim. On the symmetric relation `
    + `TransE scores <span class="value">${sym.t.toFixed(3)}</span>, RotatE `
    + `<span class="value">${sym.r.toFixed(3)}</span> and DistMult `
    + `<span class="value">${sym.d.toFixed(3)}</span>`
    + `${dSym
      ? ': DistMult is symmetric by construction, so this is the one column where being unable to '
        + 'tell a direction apart costs nothing, and it does not.'
      : ': DistMult is symmetric by construction, so this is the column where that should cost it '
        + 'nothing, and here it still trails the other two, which is a fact about this fit and not '
        + 'about the scoring function.'} `
    + `On the hierarchy, which has a direction, DistMult `
    + `${dDrops
      ? `falls from that <span class="value">${sym.d.toFixed(3)}</span> to `
        + `<span class="value">${hier.d.toFixed(3)}</span>`
      : `scores <span class="value">${hier.d.toFixed(3)}</span>`}, `
    + `because \\(\\langle h, r, t\\rangle\\) is the same number with the ends swapped and it `
    /* the comparison that carries the claim is DistMult against ITSELF on the
       two columns, not against the other two models: on this run it sits above
       TransE on the hierarchy as well, and quoting the three side by side made
       the sentence read as if it came last */
    + `cannot prefer one order. `
    + `${hier.d > Math.min(hier.t, hier.r)
      ? `It is not last in that column, mind: TransE manages `
        + `<span class="value">${hier.t.toFixed(3)}</span> and RotatE `
        + `<span class="value">${hier.r.toFixed(3)}</span>, so being able to represent a direction `
        + `is not the same as finding it at this budget. `
      : `TransE reaches <span class="value">${hier.t.toFixed(3)}</span> and RotatE `
        + `<span class="value">${hier.r.toFixed(3)}</span>, both above it. `}`
    + `And on the composition `
    + `the two movement models do <span class="value">${comp.t.toFixed(3)}</span> and `
    + `<span class="value">${comp.r.toFixed(3)}</span>`
    + `${compFree
      ? `, against ${comp.d.toFixed(3)}: composing two relations is adding two vectors or `
        + 'multiplying two rotations, which is something the algebra hands them for free.'
      : `, against ${comp.d.toFixed(3)} for DistMult, so composition did not separate them here `
        + 'the way adding two vectors says it should.'}`;

  return { patterns };
}
