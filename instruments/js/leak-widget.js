/* The leak divided by the lever, as a grid.
 *
 * A grid rather than a set of curves because the claim is multiplicative: the
 * damage is not a function of the leak or of the lever but of their ratio, and
 * that reads off a rectangle in one look and off five curves in none.
 */
import { makeChart, annotate } from '../../assets/js/chart.js';

const f2 = (v) => v.toFixed(2);
const f3 = (v) => v.toFixed(3);

export function initLeakWidget(data) {
  const host = document.querySelector('#leak-widget');
  if (!host) return;
  const M = data.meta;
  const E = data.exclusion;
  const chart = makeChart('#leak-chart', {
    width: 640, height: 420, margin: { top: 52, right: 30, bottom: 62, left: 78 },
  });
  const { g, w, h } = chart;
  const readout = document.querySelector('#leak-readout');
  const views = d3.select('#leak-views');
  let view = 'measured';

  const buttons = [
    ['measured', 'what the estimator returned'],
    ['closed', 'the leak over the lever'],
    ['gap', 'the difference between them'],
  ];
  views.selectAll('button')
    .data(buttons)
    .join('button')
    .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'))
    .text((d) => d[1])
    .on('click', (_, d) => { view = d[0]; render(); });

  const kappas = E[0].cells.map((c) => c.kappa);
  const pis = E.map((r) => r.pi);
  const accent = getComputedStyle(document.documentElement)
    .getPropertyValue('--primary').trim() || '#056605';

  function value(row, cell) {
    if (view === 'measured') return cell.bias;
    if (view === 'closed') return cell.closed;
    return cell.bias - cell.closed;
  }

  function render() {
    views.selectAll('button')
      .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'));
    g.selectAll('*').remove();
    const x = d3.scaleBand().domain(kappas.map(f2)).range([0, w]).padding(0.06);
    const y = d3.scaleBand().domain(pis.map(f2)).range([0, h]).padding(0.06);
    const all = E.flatMap((row) => row.cells.map((c) => Math.abs(value(row, c))));
    const scale = d3.scaleSequential(d3.interpolateLab('#f1f3f3', accent))
      .domain([0, Math.max(...all) || 1]);

    E.forEach((row) => {
      row.cells.forEach((c) => {
        const v = value(row, c);
        g.append('rect')
          .attr('x', x(f2(c.kappa))).attr('y', y(f2(row.pi)))
          .attr('width', x.bandwidth()).attr('height', y.bandwidth())
          .attr('fill', scale(Math.abs(v)))
          .attr('stroke', 'var(--paper)').attr('stroke-width', 1.4);
        const dark = Math.abs(v) > 0.55 * (Math.max(...all) || 1);
        g.append('text')
          .attr('x', x(f2(c.kappa)) + x.bandwidth() / 2)
          .attr('y', y(f2(row.pi)) + y.bandwidth() / 2 + 4)
          .attr('text-anchor', 'middle')
          .style('font-size', '12.5px')
          .attr('fill', dark ? 'var(--paper)' : 'var(--squidink)')
          .text(view === 'gap' ? v.toFixed(3) : f2(v));
      });
    });
    kappas.forEach((k) => {
      g.append('text').attr('class', 'chart-note')
        .attr('x', x(f2(k)) + x.bandwidth() / 2).attr('y', h + 20)
        .attr('text-anchor', 'middle').style('font-size', '12px')
        .text(f2(k));
    });
    pis.forEach((p) => {
      g.append('text').attr('class', 'chart-note')
        .attr('x', -10).attr('y', y(f2(p)) + y.bandwidth() / 2 + 4)
        .attr('text-anchor', 'end').style('font-size', '12px')
        .text(f2(p));
    });
    annotate(g, w / 2, h + 46, 'the direct path from the lever to the outcome',
      { anchor: 'middle' }).style('font-size', '12px');
    g.append('text').attr('class', 'chart-annotation')
      .attr('transform', `rotate(-90) translate(${-h / 2}, -52)`)
      .attr('text-anchor', 'middle').style('font-size', '12px')
      .text('how hard the lever pushes');
    annotate(g, 0, -14, {
      measured: 'bias of the ratio, measured over 120 samples per cell',
      closed: 'the closed form: the leak divided by the lever',
      gap: 'measured minus closed form',
    }[view], { anchor: 'start' }).style('font-size', '12px');

    const weak = E[E.length - 1];
    const cell = weak.cells.find((c) => c.kappa === 0.05) ?? weak.cells[2];
    const strong = E[0].cells.find((c) => c.kappa === 0.05) ?? E[0].cells[2];
    const zero = weak.cells.find((c) => c.kappa === 0);
    readout.innerHTML =
      `<p>Read the top row against the bottom one. A direct path of `
      + `${f2(cell.kappa)} is a small thing: it is ${f2(cell.kappa / M.beta * 100)}% of the `
      + `effect being estimated, the kind of imperfection an instrument gets defended with. `
      + `With a strong lever it costs <span class="bold">${f3(strong.bias)}</span>. With a `
      + `lever of ${f2(weak.pi)} the same path costs `
      + `<span class="bold">${f3(cell.bias)}</span>, and the closed form says `
      + `${f3(cell.closed)}. ${zero ? `The first column is the control: with no leak at all `
        + `that row is still off by ${f3(zero.bias)}, which is the weak instrument bias from `
        + `the previous section, so the rest of the row has to be read against that rather `
        + `than against zero.` : ''} An instrument that is both weak and slightly `
      + `contaminated is not slightly worse than a good one.</p>`;
  }

  render();
}
