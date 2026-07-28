/* The ceiling, as a picture.
 *
 * There is no data in this widget at all, which is the point: the largest
 * z-score a sample can produce depends on the size of the sample and on how
 * many of its members are odd together, and on nothing else. Everything drawn
 * is the closed form, and the guard checks it against a generator that got the
 * same numbers by pushing points to 1e9 and reading the answer off.
 */
import { makeChart, drawAxes, axisLabels, drawGrid } from '../../assets/js/chart.js';
import { zCeiling } from './statkit.js';
import { seriesLabel, legend } from './draw.js';

export function initCeilingWidget(data) {
  const Z_CUT = data.one_d.z_cut;
  const N_WINE = data.many_d.n;
  const N_SUB = data.one_d.sub.n;
  const K_GRID = data.one_d.ceiling.k_rows.map((r) => r.k);

  let k = 1;
  const slider = d3.select('#ceil-k')
    .attr('min', 0).attr('max', K_GRID.length - 1).attr('step', 1).property('value', 0);
  const readout = d3.select('#ceil-readout');

  const chart = makeChart('#ceil-chart', {
    width: 640, height: 340, margin: { top: 46, right: 116, bottom: 56, left: 58 },
  });
  const { g, w, h } = chart;

  const N_MIN = 4;
  const N_MAX = 1000;
  const x = d3.scaleLog().domain([N_MIN, N_MAX]).range([0, w]);
  /* Root scale, and no clamping. Clamping the curve to the top of the panel
     drew a flat ceiling that does not exist, which in a widget about a real
     ceiling is the one lie that must not be told. The largest value anything
     here reaches is one odd reading in a thousand. */
  const Y_MAX = zCeiling(N_MAX, 1) * 1.03;
  const yVals = [0, 1, 2, 3, 5, 8, 12, 18, 25, 32].filter((v) => v <= Y_MAX);
  const y = d3.scaleSqrt().domain([0, Y_MAX]).range([h, 0]);
  const nGrid = d3.range(0, 401).map((i) => N_MIN * (N_MAX / N_MIN) ** (i / 400));
  const line = d3.line().x((d) => x(d.n)).y((d) => y(d.v));

  const GHOSTS = [1, 2, 3, 5, 10, 20];

  function render() {
    const kk = K_GRID[+slider.property('value')];
    k = kk;
    g.selectAll('*').remove();

    drawGrid(g, x, y, w, h, { xValues: [10, 100, 1000], yValues: yVals });
    drawAxes(g, x, y, w, h, { xValues: [10, 100, 1000], xFmt: d3.format(','), yValues: yVals });
    axisLabels(g, w, h, { x: 'how many readings there are', y: 'largest possible z, root scale' });

    /* The band below the threshold: any curve lying inside it belongs to a
       sample size at which the rule cannot fire at all, whatever the data. */
    g.append('rect')
      .attr('x', 0).attr('y', y(Z_CUT)).attr('width', w).attr('height', h - y(Z_CUT))
      .attr('fill', 'var(--cosmos)').attr('opacity', 0.07);
    g.append('line')
      .attr('x1', 0).attr('x2', w).attr('y1', y(Z_CUT)).attr('y2', y(Z_CUT))
      .attr('stroke', 'var(--cosmos)').attr('stroke-width', 1.4).attr('stroke-dasharray', '4 3');
    seriesLabel(g, w + 6, y(Z_CUT) + 4, 'three sigma', { size: 10.5, opacity: 0.9 })
      .style('fill', 'var(--cosmos)');
    /* No caption inside the band. There is no corner of it that stays empty at
       every setting of the slider (the bottom right holds the twelve-bottle dot
       at eight odd readings, the strip under the line holds the 178 dot between
       seven and thirteen), and a caption that collides with a mark at four
       settings out of forty is worse than no caption. What the band means is
       carried by the labelled threshold above it and spelled out in the
       sentence below the picture. */

    GHOSTS.filter((q) => q !== kk).forEach((q) => {
      g.append('path')
        .attr('d', line(nGrid.filter((n) => n > q).map((n) => ({ n, v: zCeiling(n, q) }))))
        .attr('fill', 'none').attr('stroke', 'var(--squidink)')
        .attr('stroke-width', 1).attr('opacity', 0.2);
    });
    g.append('path')
      .attr('d', line(nGrid.filter((n) => n > kk).map((n) => ({ n, v: zCeiling(n, kk) }))))
      .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2.4);

    /* the two sample sizes this article actually uses. The labels go below and
       to the left of their dots: above them is where the curve is, and to the
       right of the second one is the edge of the canvas. */
    [{ n: N_SUB, label: `the ${N_SUB} bottles above` },
      { n: N_WINE, label: `all ${N_WINE} bottles` }].forEach((m) => {
      if (m.n <= kk) return;
      const v = zCeiling(m.n, kk);
      const px = x(m.n);
      const py = y(v);
      g.append('circle').attr('cx', px).attr('cy', py).attr('r', 4.5)
        .attr('fill', 'var(--primary)').attr('stroke', 'var(--paper)').attr('stroke-width', 1.4);
      const right = px > 0.55 * w;
      /* Clear of the three-sigma rule as well as of the curve: at the counts
         where the two nearly coincide, "below the dot" and "on the dashed
         line" are the same place. */
      const dy = Math.abs(py - y(Z_CUT)) < 18 ? 30 : 16;
      seriesLabel(g, px + (right ? -9 : 9), py + dy, `${m.label}: ${v.toFixed(3)}`,
        { size: 10.5, opacity: 0.95, anchor: right ? 'end' : 'start' });
    });

    const chips = legend(g, [
      { label: `${kk} odd reading${kk === 1 ? '' : 's'} together`, shape: 'line', colour: 'var(--primary)' },
      { label: `the other counts (${GHOSTS.filter((q) => q !== kk).join(', ')})`,
        shape: 'line', colour: 'var(--stone)' },
    ], { x0: 2, y0: -26, gap: 240, size: 10.5 });
    /* the ghosts are drawn faint, so their chip is too, or the legend promises
       a line that is not on the picture */
    chips.filter((d, i) => i === 1).select('path.mark').attr('stroke-opacity', 0.75);

    /* the smallest sample in which three sigma is arithmetically possible */
    let first = null;
    for (let n = kk + 1; n <= 200000; n++) {
      if (zCeiling(n, kk) >= Z_CUT) { first = n; break; }
    }
    d3.select('#ceil-k-label').text(kk);
    readout.html(
      `<span class="slider-label">with <span class="value">${kk}</span> odd reading`
      + `${kk === 1 ? ' in the sample' : 's sitting together'}, three sigma is out of reach until there are `
      + `<span class="value">${first === null ? 'never' : first.toLocaleString('en-US')}</span> `
      + `readings in the sample.</span> `
      + (N_SUB > kk
        ? `At the ${N_SUB} bottles the bench above uses the ceiling is `
          + `${zCeiling(N_SUB, kk).toFixed(3)}, and at all ${N_WINE} it is `
          + `${zCeiling(N_WINE, kk).toFixed(3)}. `
        : `The ${N_SUB} bottles of the bench above cannot hold ${kk} odd readings at all, so the `
          + `curve starts to the right of them; at all ${N_WINE} the ceiling is `
          + `${zCeiling(N_WINE, kk).toFixed(3)}. `)
      + `Nothing in this picture knows anything about wine: push the odd readings to a million or `
      + `to a trillion and the answer does not move, because the distance cancels out of the ratio.`
    );
  }

  slider.on('input', render);
  render();
  return { render };
}
