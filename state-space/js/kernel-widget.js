/* Three initialisations, three kernels, one log axis.
 *
 * The table below this widget has the numbers and is the record. What a table
 * cannot show is that these three curves are not three points on a spectrum:
 * two of them fall and one of them climbs, and no amount of training fixes the
 * third, because a multiplier above one raised to the power of the sequence
 * length is what it is.
 *
 * The vertical range is deliberately clamped to four decades either side of
 * one. The unstable kernel reaches 1e+75, and a chart with room for that would
 * squash the two useful curves into a line one pixel from the floor. So the
 * divergent one is allowed to leave through the top and is labelled with where
 * it actually ends up, which is more honest than a chart nobody can read and
 * more informative than leaving it out.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { seriesLabel, logTicks } from '../../assets/js/textdraw.js';

const LO = 1e-4;
const HI = 1e4;

export function initKernelWidget(data) {
  const R = data.reach;
  if (!R || !R.length) return null;
  let pick = R.findIndex((r) => !r.unstable && r.hundredth_at !== null);
  if (pick < 0) pick = 0;

  const chart = makeChart('#kernel-chart', {
    width: 640, height: 340, margin: { top: 40, right: 96, bottom: 52, left: 66 },
    clip: true,
  });
  const { g, plot, w, h } = chart;
  const readout = d3.select('#kernel-readout');

  const L = Math.max(...R.map((r) => r.kernel.length));
  const x = d3.scaleLinear().domain([0, L - 1]).range([0, w]);
  const y = d3.scaleLog().domain([LO, HI]).range([h, 0]);
  const clampY = (v) => y(Math.min(HI * 4, Math.max(LO / 4, Math.abs(v) || LO / 4)));

  d3.select('#kernel-inits').selectAll('button').data(R).join('button')
    .attr('class', (d, i) => `atlas-btn${i === pick ? '' : ' ghost'}`)
    .text((d) => d.init)
    .on('click', (event, d, i) => {
      pick = R.indexOf(d);
      d3.select('#kernel-inits').selectAll('button')
        .attr('class', (q) => `atlas-btn${R.indexOf(q) === pick ? '' : ' ghost'}`);
      render();
    });

  function render() {
    g.selectAll('*').remove();
    plot.selectAll('*').remove();
    drawGrid(g, x, y, w, h, { xTicks: 6, yValues: logTicks(LO, HI) });

    /* the hundredth line, which is what the table's third column measures */
    g.append('line')
      .attr('x1', 0).attr('x2', w).attr('y1', y(0.01)).attr('y2', y(0.01))
      .attr('stroke', 'var(--squidink)').attr('stroke-width', 1)
      .attr('stroke-dasharray', '3,3').attr('opacity', 0.4);
    seriesLabel(g, w - 4, y(0.01) - 6, 'a hundredth of where it started',
      { anchor: 'end', size: 10, opacity: 0.55 });

    const line = d3.line().x((d, t) => x(t)).y((d) => clampY(d));
    R.forEach((r, i) => {
      const on = i === pick;
      plot.append('path').datum(r.kernel).attr('d', line)
        .attr('fill', 'none')
        .attr('stroke', r.unstable ? 'var(--secondary)' : 'var(--primary)')
        .attr('stroke-width', on ? 2.6 : 1.4)
        .attr('stroke-dasharray', r.unstable ? '5,3' : null)
        .attr('opacity', on ? 1 : 0.32);

      /* where it crosses the hundredth, for the ones that do */
      if (on && r.hundredth_at !== null && r.hundredth_at < L) {
        g.append('circle')
          .attr('cx', x(r.hundredth_at)).attr('cy', y(0.01)).attr('r', 5)
          .attr('fill', 'var(--paper)').attr('stroke', 'var(--primary)')
          .attr('stroke-width', 2.2);
        seriesLabel(g, x(r.hundredth_at) + 8, y(0.01) + 16,
          `${r.hundredth_at} steps`, { size: 11, weight: 600 });
      }
    });

    /* the one that leaves through the top gets told where it went */
    const bad = R.find((r) => r.unstable);
    if (bad) {
      const exit = bad.kernel.findIndex((v) => Math.abs(v) > HI);
      const ex = exit >= 0 ? exit : L - 1;
      seriesLabel(g, Math.min(w - 6, x(ex) + 6), 14,
        `off the top at step ${ex}`, { anchor: 'start', size: 10.5, opacity: 0.8 });
    }

    drawAxes(g, x, y, w, h, {
      xTicks: 6,
      yValues: logTicks(LO, HI),
      yFmt: (v) => (v >= 1 ? d3.format('~s')(v) : d3.format('.0e')(v)),
    });
    axisLabels(g, w, h, { x: 'steps back', y: 'kernel, relative to step zero' });

    const r = R[pick];
    readout.html(
      `<div><span class="bold">${r.init}</span>: the largest multiplier in the diagonal is `
      + `<span class="value">${r.max_abs_a.toFixed(4)}</span>`
      + (r.unstable
        ? `, which is above one, so the kernel does not reach further, it <span class="bold">`
          + `diverges</span>: by the end of ${L} steps it is `
          + `<span class="value">${r.growth.toExponential(2)}</span> times its starting size. `
          + `Nothing downstream can recover from that, and no amount of training moves a `
          + `multiplier past the exponent it is raised to.</div>`
        : `, and the kernel is down to a hundredth of its starting size after `
          + `<span class="value">${r.hundredth_at === null ? `more than ${L}` : r.hundredth_at}</span>`
          + ` steps. That number is this model's version of the half-life the recurrent article `
          + `measured on the gradient, and here it is decided entirely before training starts.</div>`)
      + '<div>All three lines are drawn at once so the comparison is one glance: the dashed one '
      + 'is unstable and leaves the top of the chart, and the two solid ones differ only in how '
      + 'the same decays were spread. Spreading them on a logarithmic scale is the whole content '
      + 'of the structured initialisations these models are named after.</div>'
    );
  }

  render();
  return { render };
}
