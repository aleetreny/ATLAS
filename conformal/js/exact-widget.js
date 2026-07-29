/* The coverage, and the distribution the coverage is an average of.
 *
 * The Beta curve is the point of this widget. A reader who takes away only
 * "the coverage is exactly ninety percent" has taken away a true sentence and
 * the wrong idea, and the only way to fix that is to draw the thing the
 * average is over.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, annotate } from '../../assets/js/chart.js';
import { rankFor, betaPdf } from './conformalkit.js';

const n0 = (v) => v.toLocaleString('en-US');
const f3 = (v) => v.toFixed(3);
const f6 = (v) => v.toFixed(6);
const pc = (v, d = 1) => `${(100 * v).toFixed(d)}%`;

export function initExactWidget(data) {
  const host = document.querySelector('#exact-widget');
  if (!host) return;
  const M = data.meta;
  const X = data.exact;
  const chart = makeChart('#exact-chart', {
    width: 640, height: 400, margin: { top: 44, right: 30, bottom: 58, left: 70 },
  });
  const { g, w, h } = chart;
  const readout = document.querySelector('#exact-readout');
  const label = document.querySelector('#exact-n-label');
  const slider = document.querySelector('#exact-n');
  const views = d3.select('#exact-views');
  let view = 'spread';
  let i = 2;

  const buttons = [
    ['spread', 'the spread between calibration sets'],
    ['formula', 'the formula against the target'],
  ];
  views.selectAll('button')
    .data(buttons)
    .join('button')
    .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'))
    .text((d) => d[1])
    .on('click', (_, d) => { view = d[0]; render(); });

  function render() {
    views.selectAll('button')
      .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'));
    g.selectAll('*').remove();
    const row = X.rows[i];
    label.textContent = n0(row.n);
    const line = d3.line().x((d) => d[0]).y((d) => d[1]);

    if (view === 'spread') {
      const lo = Math.max(0.55, row.formula - 6 * row.sd_closed);
      const hi = Math.min(1, row.formula + 6 * row.sd_closed);
      const xs = d3.range(lo, hi, (hi - lo) / 400);
      const dens = xs.map((v) => betaPdf(v, row.beta_a, row.beta_b));
      const x = d3.scaleLinear().domain([lo, hi]).range([0, w]);
      const y = d3.scaleLinear().domain([0, d3.max(dens) * 1.12]).range([h, 0]);
      drawGrid(g, x, y, w, h);
      drawAxes(g, x, y, w, h, { xFmt: d3.format('.0%'), yValues: [] });
      axisLabels(g, w, h, { x: 'coverage this calibration set actually delivers' });

      /* The measured histogram, only where it exists: it was recorded at one
       * size, so it is drawn at that size and not implied at the others. */
      if (row.n === X.hist.n) {
        const edges = X.hist.edges;
        g.append('g').selectAll('rect')
          .data(X.hist.density)
          .join('rect')
          .attr('x', (_, k) => x(edges[k]))
          .attr('width', (_, k) => Math.max(0.5, x(edges[k + 1]) - x(edges[k])))
          .attr('y', (d) => y(d))
          .attr('height', (d) => h - y(d))
          .attr('fill', 'var(--stone)');
      }
      g.append('path').attr('d', line(xs.map((v, k) => [x(v), y(dens[k])])))
        .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2.6);
      const target = 1 - M.alpha;
      g.append('line').attr('x1', x(target)).attr('x2', x(target))
        .attr('y1', 0).attr('y2', h)
        .attr('stroke', 'var(--anchor)').attr('stroke-width', 1.8)
        .attr('stroke-dasharray', '5 4');
      annotate(g, x(target) - 6, 14, `asked for ${pc(target, 0)}`, { anchor: 'end' })
        .style('font-size', '12px');
      const below = d3.range(lo, target, (target - lo) / 200);
      g.append('path')
        .attr('d', `${line(below.map((v) => [x(v), y(betaPdf(v, row.beta_a, row.beta_b))]))}L${x(target)},${y(0)}L${x(lo)},${y(0)}Z`)
        .attr('fill', 'var(--cosmos)').attr('fill-opacity', 0.28);
      annotate(g, 4, -14, row.n === X.hist.n
        ? 'bars: twenty thousand calibration sets. curve: the exact Beta'
        : 'curve: the exact Beta for this calibration size', { anchor: 'start' })
        .style('font-size', '12px');

      readout.innerHTML =
        `<p>With <span class="bold">${n0(row.n)}</span> calibration points the procedure `
        + `uses the ${n0(row.k)}-th smallest score, and its coverage averaged over `
        + `calibration sets is <span class="bold">${f6(row.formula)}</span>. Any one `
        + `calibration set is a draw from the shaded curve, which is a Beta with parameters `
        + `${n0(row.beta_a)} and ${n0(row.beta_b)}: its spread is `
        + `<span class="bold">${f3(row.sd)}</span> measured and ${f3(row.sd_closed)} in `
        + `closed form, and <span class="bold">${pc(row.below)}</span> of calibration sets `
        + `land in the red region, below what was asked for. The fifth and ninety fifth `
        + `percentiles are ${pc(row.q05)} and ${pc(row.q95)}. Slide to ${n0(X.rows[X.rows.length - 1].n)} `
        + `points and the curve collapses onto the line.</p>`;
    } else {
      const x = d3.scaleLog().domain([X.rows[0].n, X.rows[X.rows.length - 1].n]).range([0, w]);
      const y = d3.scaleLinear().domain([1 - M.alpha - 0.002, 1 - M.alpha + 0.012])
        .range([h, 0]);
      const xt = X.rows.map((r) => r.n);
      drawGrid(g, x, y, w, h, { xValues: xt });
      drawAxes(g, x, y, w, h, { xValues: xt, yFmt: d3.format('.1%') });
      axisLabels(g, w, h, { x: 'calibration points', y: 'coverage' });
      g.append('line').attr('x1', 0).attr('x2', w)
        .attr('y1', y(1 - M.alpha)).attr('y2', y(1 - M.alpha))
        .attr('stroke', 'var(--anchor)').attr('stroke-width', 1.6)
        .attr('stroke-dasharray', '5 4');
      annotate(g, 4, y(1 - M.alpha) - 8, `asked for ${pc(1 - M.alpha, 0)}`, { anchor: 'start' })
        .style('font-size', '11.5px');
      g.append('path')
        .attr('d', line(X.rows.map((r) => [x(r.n), y(r.upper)])))
        .attr('fill', 'none').attr('stroke', 'var(--stone)').attr('stroke-width', 2)
        .attr('stroke-dasharray', '4 4');
      g.append('path')
        .attr('d', line(X.rows.map((r) => [x(r.n), y(r.formula)])))
        .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2.6);
      X.rows.forEach((r) => {
        g.append('circle').attr('cx', x(r.n)).attr('cy', y(r.mean)).attr('r', 4)
          .attr('fill', 'var(--cosmos)');
      });
      g.append('line').attr('x1', x(row.n)).attr('x2', x(row.n)).attr('y1', 0).attr('y2', h)
        .attr('stroke', 'var(--stone)').attr('stroke-width', 1.4);
      annotate(g, w - 4, -28, 'line: the formula', { anchor: 'end' })
        .style('font-size', '11.5px').attr('fill', 'var(--primary)');
      annotate(g, w - 4, -14, 'dots: twenty thousand calibration sets each', { anchor: 'end' })
        .style('font-size', '11.5px').attr('fill', 'var(--cosmos)');
      annotate(g, w - 4, 0, 'dashed: the upper bound the theorem also gives', { anchor: 'end' })
        .style('font-size', '11.5px');
      readout.innerHTML =
        `<p>The coverage is never below the target and never far above it: it sits between `
        + `\\(1-\\alpha\\) and \\(1-\\alpha+1/(n+1)\\), which is the pair of lines drawn `
        + `here, and the dots are the measurements. At `
        + `<span class="bold">${n0(row.n)}</span> points the formula says `
        + `${f6(row.formula)} and ${n0(X.trials)} calibration sets deliver `
        + `<span class="bold">${f6(row.mean)}</span>. The excess is `
        + `${((row.formula - (1 - M.alpha)) * 100).toFixed(4)} percentage points, which is `
        + `the price of a discrete rank: with ${n0(row.n)} scores there is no cut that gives `
        + `exactly ${pc(1 - M.alpha, 0)}, so the method takes the smallest one that is `
        + `safely above.</p>`;
    }
  }

  slider.addEventListener('input', (e) => { i = +e.target.value; render(); });
  render();
}
