/* The balancing property, and its control.
 *
 * Four views of the same four bars. The one that matters is the third: a score
 * built from half the covariates balances that half and leaves the rest where
 * they were, which is the difference between "the score removes confounding"
 * and "the score removes confounding by the variables you gave it".
 */
import { makeChart, drawAxes, drawGrid, axisLabels, annotate } from '../../assets/js/chart.js';

const f3 = (v) => v.toFixed(3);

export function initBalanceWidget(data) {
  const host = document.querySelector('#balance-widget');
  if (!host) return;
  const B = data.balance;
  const chart = makeChart('#balance-chart', {
    width: 640, height: 400, margin: { top: 36, right: 26, bottom: 96, left: 66 },
  });
  const { g, w, h } = chart;
  const readout = document.querySelector('#balance-readout');
  const views = d3.select('#balance-views');
  let view = 'before';

  const buttons = [
    ['before', 'as collected'],
    ['true', 'inside strata of the true score'],
    ['wrong', 'a score missing two covariates'],
    ['ipw', 'weighted by the true score'],
  ];
  const series = {
    before: B.before, true: B.after_true, wrong: B.after_wrong, ipw: B.after_ipw,
  };
  views.selectAll('button')
    .data(buttons)
    .join('button')
    .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'))
    .text((d) => d[1])
    .on('click', (_, d) => { view = d[0]; render(); });

  const short = ['V0', 'V1', 'V2', 'V3'];
  /* Los nombres largos de la exportación ("treatment and outcome") miden lo
   * mismo para V0 y para V1 y se pisaban entre columnas contiguas. Se acortan
   * aquí, donde se sabe el ancho de la banda. */
  const roles = B.cols.map((c) => (/treatment and outcome/.test(c) ? 'both'
    : /outcome only/.test(c) ? 'outcome' : 'treatment'));

  function render() {
    views.selectAll('button')
      .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'));
    g.selectAll('*').remove();
    const vals = series[view];
    const x = d3.scaleBand().domain(short).range([0, w]).padding(0.34);
    const lim = Math.max(0.95, d3.max(B.before.map(Math.abs)) * 1.08);
    const y = d3.scaleLinear().domain([-lim, lim]).range([h, 0]);
    drawGrid(g, x, y, w, h);
    drawAxes(g, x, y, w, h);
    axisLabels(g, w, h, { y: 'standardised difference', xOffset: 66 });

    /* The conventional line for "balanced", drawn on both sides because a
     * standardised difference has a sign and the reader is being asked to
     * compare magnitudes. */
    [-0.1, 0.1].forEach((v) => {
      g.append('line')
        .attr('x1', 0).attr('x2', w).attr('y1', y(v)).attr('y2', y(v))
        .attr('stroke', 'var(--anchor)').attr('stroke-width', 1.3)
        .attr('stroke-dasharray', '4 4');
    });
    /* Fuera del área de barras: dentro caía encima de una etiqueta de valor. */
    annotate(g, w - 4, -14, 'dashed: the conventional 0.1', { anchor: 'end' })
      .style('font-size', '11.5px');

    g.append('g').selectAll('rect')
      .data(vals)
      .join('rect')
      .attr('x', (_, i) => x(short[i]))
      .attr('y', (d) => (d >= 0 ? y(d) : y(0)))
      .attr('width', x.bandwidth())
      .attr('height', (d) => Math.abs(y(d) - y(0)))
      .attr('fill', (d) => (Math.abs(d) > 0.1 ? 'var(--cosmos)' : 'var(--primary)'));
    g.append('g').selectAll('text.val')
      .data(vals)
      .join('text')
      .attr('class', 'chart-note')
      .attr('x', (_, i) => x(short[i]) + x.bandwidth() / 2)
      .attr('y', (d) => (d >= 0 ? y(d) - 7 : y(d) + 15))
      .attr('text-anchor', 'middle')
      .style('font-size', '12px')
      .text((d) => f3(d));
    g.append('g').selectAll('text.role')
      .data(roles)
      .join('text')
      .attr('class', 'chart-note')
      .attr('x', (_, i) => x(short[i]) + x.bandwidth() / 2)
      .attr('y', h + 34)
      .attr('text-anchor', 'middle')
      .style('font-size', '11.5px')
      .text((d) => d.replace(' and ', ' + '));

    const worst = Math.max(...vals.map(Math.abs));
    const notes = {
      before: `As collected, the treated and the untreated are different people on `
        + `${vals.filter((v) => Math.abs(v) > 0.1).length} of the four covariates, the worst `
        + `at <span class="bold">${f3(worst)}</span>. V2 is already balanced because nothing `
        + `about the treatment depends on it, which is what makes it a useful control: it `
        + `should stay balanced whatever anybody does.`,
      true: `Inside thin strata of the true score, every covariate falls under the `
        + `conventional line, the worst at <span class="bold">${f3(worst)}</span>. Nobody `
        + `matched on the covariates. They were matched on one number computed from them, `
        + `and that is the theorem.`,
      wrong: `This score was built from V0 and V2 only. It balances V0 to `
        + `${f3(Math.abs(vals[0]))} and leaves <span class="bold">V1 at ${f3(vals[1])} and `
        + `V3 at ${f3(vals[3])}</span>, essentially where they started. If the two missing `
        + `covariates had never been recorded, this table would be the balance table in the `
        + `paper, and it would show two columns and both of them fine.`,
      ipw: `Weighting rather than stratifying: each unit counted by the inverse of the `
        + `probability of the treatment it got. Worst standardised difference `
        + `<span class="bold">${f3(worst)}</span>, and the signs are now on the other side of `
        + `zero, which is what a reweighted pseudo-population looks like when the weights `
        + `are doing their job.`,
    };
    readout.innerHTML = `<p>${notes[view]}</p>`;
  }

  render();
}
