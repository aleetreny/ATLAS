/* The shape of the problem, before any weighting exists.
 *
 * Five states over the same rank-frequency data. The first is linear on both
 * axes, which is the honest way to show why nobody plots it that way: the
 * curve is a wall and a floor and there is nothing to read. Everything after
 * that is the same numbers on log axes.
 *
 * The last step swaps the series for the vocabulary growth curve, which is a
 * different measurement of the same corpus and answers the question the fourth
 * step raises: if the tail is that long, does it ever stop.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { scrolly } from '../../assets/js/scrolly.js';
import { seriesLabel, legend, logTicks, shortNum } from '../../assets/js/textdraw.js';

export function initScrollyZipf(data) {
  const Z = data.zipf;
  const H = data.heaps;
  const chart = makeChart('#zipf-scrolly-chart', {
    width: 640, height: 440, margin: { top: 42, right: 26, bottom: 58, left: 66 },
  });
  const { g, w, h } = chart;

  const curve = Z.curve.map(([r, c]) => ({ r, c }));
  const heapsCurve = H.curve.map(([t, v]) => ({ t, v }));

  const layers = {
    grid: g.append('g').attr('class', 'grid'),
    axes: g.append('g'),
    marks: g.append('g'),
    notes: g.append('g'),
  };

  function clear() {
    layers.grid.selectAll('*').remove();
    layers.axes.selectAll('*').remove();
    layers.marks.selectAll('*').remove();
    layers.notes.selectAll('*').remove();
  }

  /* ---------------------------------------------------------------- linear */
  function drawLinear() {
    clear();
    const x = d3.scaleLinear().domain([0, Z.types]).range([0, w]);
    const y = d3.scaleLinear().domain([0, curve[0].c]).range([h, 0]);
    drawGrid(layers.grid, x, y, w, h, { xTicks: 5, yTicks: 5 });
    drawAxes(layers.axes, x, y, w, h, { xTicks: 5, yTicks: 5, xFmt: shortNum, yFmt: shortNum });
    axisLabels(layers.axes, w, h, { x: 'rank of the word', y: 'times it occurs' });
    layers.marks.append('path')
      .datum(curve)
      .attr('fill', 'none')
      .attr('stroke', 'var(--primary)')
      .attr('stroke-width', 2)
      .attr('d', d3.line().x((d) => x(d.r)).y((d) => y(d.c)));
    seriesLabel(layers.notes, w - 4, 22, 'both axes linear', { anchor: 'end', size: 11.5 });
  }

  /* ------------------------------------------------------------------- log */
  function logScales() {
    const x = d3.scaleLog().domain([1, Z.types]).range([0, w]);
    const y = d3.scaleLog().domain([0.8, curve[0].c]).range([h, 0]);
    return { x, y };
  }

  function drawLog({ fit = false, hapax = false, cover = false } = {}) {
    clear();
    const { x, y } = logScales();
    const xv = logTicks(1, Z.types);
    const yv = logTicks(1, curve[0].c);
    drawGrid(layers.grid, x, y, w, h, { xValues: xv, yValues: yv });
    drawAxes(layers.axes, x, y, w, h, { xValues: xv, yValues: yv, xFmt: shortNum, yFmt: shortNum });
    axisLabels(layers.axes, w, h, { x: 'rank of the word', y: 'times it occurs' });

    if (hapax) {
      const first = curve.find((d) => d.c === 1);
      if (first) {
        layers.marks.append('rect')
          .attr('x', x(first.r)).attr('y', 0)
          .attr('width', Math.max(0, w - x(first.r))).attr('height', h)
          .attr('fill', 'var(--cosmos)').attr('opacity', 0.1);
      }
    }
    if (cover) {
      layers.marks.append('rect')
        .attr('x', 0).attr('y', 0)
        .attr('width', x(Z.types_for_half)).attr('height', h)
        .attr('fill', 'var(--primary)').attr('opacity', 0.12);
    }

    layers.marks.append('path')
      .datum(curve)
      .attr('fill', 'none')
      .attr('stroke', 'var(--primary)')
      .attr('stroke-width', 2)
      .attr('d', d3.line().x((d) => x(d.r)).y((d) => y(Math.max(d.c, 0.8))));

    if (fit) {
      const pts = [Z.fit_lo, Z.fit_hi].map((r) => ({
        r, c: 10 ** (Z.intercept + Z.slope * Math.log10(r)),
      }));
      layers.marks.append('line')
        .attr('x1', x(pts[0].r)).attr('y1', y(pts[0].c))
        .attr('x2', x(pts[1].r)).attr('y2', y(pts[1].c))
        .attr('stroke', 'var(--cosmos)').attr('stroke-width', 2.4)
        .attr('stroke-dasharray', '6 4');
      seriesLabel(layers.notes, x(Z.fit_hi) + 8, y(pts[1].c) - 6,
        `slope ${Z.slope.toFixed(3)}`, { anchor: 'start', size: 11.5, colour: 'var(--cosmos)' });
    }
    /* Both annotations hang off a vertical boundary, so both have to choose a
     * side. Place, ask how wide it actually came out, and flip if it does not
     * fit: the hapax band starts at 88% of the width, and a label anchored to
     * its right ran 108 units past the edge of a 640-unit canvas. Estimating
     * seven units per letter works until it does not. */
    const side = (sel, boundary, text, opts) => {
      const t = seriesLabel(sel, boundary + 6, opts.y, text,
        { anchor: 'start', size: 11.5, colour: opts.colour });
      const width = t.node().getComputedTextLength();
      if (boundary + 6 + width > w) {
        t.attr('x', boundary - 6).attr('text-anchor', 'end');
      }
      return t;
    };

    if (hapax) {
      const first = curve.find((d) => d.c === 1);
      if (first) {
        side(layers.notes, x(first.r), `${Z.hapax.toLocaleString('en-US')} words used once`,
          { y: 18, colour: 'var(--cosmos)' });
      }
    }
    if (cover) {
      side(layers.notes, x(Z.types_for_half),
        `${Z.types_for_half} words are half the text`, { y: h - 12 });
    }
  }

  /* ----------------------------------------------------------------- heaps */
  function drawHeaps() {
    clear();
    const x = d3.scaleLog().domain([Math.max(10, heapsCurve[0].t), H.final_tokens]).range([0, w]);
    const y = d3.scaleLog().domain([Math.max(5, heapsCurve[0].v), H.final_types]).range([h, 0]);
    const xv = logTicks(x.domain()[0], x.domain()[1]);
    const yv = logTicks(y.domain()[0], y.domain()[1]);
    drawGrid(layers.grid, x, y, w, h, { xValues: xv, yValues: yv });
    drawAxes(layers.axes, x, y, w, h, { xValues: xv, yValues: yv, xFmt: shortNum, yFmt: shortNum });
    axisLabels(layers.axes, w, h, { x: 'words read so far', y: 'different words seen' });
    const pts = heapsCurve.filter((d) => d.t >= x.domain()[0] && d.v >= y.domain()[0]);
    layers.marks.append('path')
      .datum(pts)
      .attr('fill', 'none')
      .attr('stroke', 'var(--primary)')
      .attr('stroke-width', 2.2)
      .attr('d', d3.line().x((d) => x(d.t)).y((d) => y(d.v)));
    const last = pts[pts.length - 1];
    layers.marks.append('circle')
      .attr('cx', x(last.t)).attr('cy', y(last.v)).attr('r', 4)
      .attr('fill', 'var(--primary)');
    legend(layers.notes, [
      { label: `still climbing at ${H.final_tokens.toLocaleString('en-US')} words`,
        colour: 'var(--primary)', shape: 'line' },
    ], { x0: 0, y0: -16, size: 11 });
    seriesLabel(layers.notes, w - 4, h - 12,
      `exponent ${H.beta.toFixed(3)}`, { anchor: 'end', size: 11.5 });
  }

  const handlers = {
    0: drawLinear,
    1: () => drawLog(),
    2: () => drawLog({ fit: true }),
    3: () => drawLog({ cover: true }),
    4: () => drawLog({ hapax: true }),
    5: drawHeaps,
  };

  drawLinear();
  const s = scrolly('#zipf-scrolly .step', handlers);
  return { ...s, facts: { types: Z.types, hapax: Z.hapax } };
}
