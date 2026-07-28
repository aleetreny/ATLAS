/* How much of the answer was in the photographs.
 *
 * The rig panel is the whole experimental design in one picture: where the
 * cameras stood, where the held-out cameras stand, and the band of elevations
 * that was photographed at all. The chart is what a model fitted to each
 * subset scores, split by whether the camera being scored is inside that band
 * or outside it, because those two numbers behave completely differently and
 * an average over both hides the finding.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, spread } from '../../assets/js/chart.js';
import { drawTiles, buttonRow, seriesLabel, fmt } from './panels.js';

export function initViewsWidget(data, sheets) {
  const meta = data.meta;
  const curve = data.views;
  const outEl = data.cameras.outside.map((c) => c.el);
  const bandLo = Math.min(...meta.rings);
  const bandHi = Math.max(...meta.rings);
  const below = outEl.map((e, i) => (e < bandLo ? i : -1)).filter((i) => i >= 0);
  const above = outEl.map((e, i) => (e > bandHi ? i : -1)).filter((i) => i >= 0);
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  const groupOf = (row, idx) => mean(idx.map((i) => row.per_outside[i]));

  const state = { k: curve[curve.length - 1].views };
  const readout = document.querySelector('#views-readout');

  const drawRig = () => {
    const node = document.querySelector('#views-rig');
    d3.select(node).selectAll('*').remove();
    /* the legend lives in the right margin rather than inside the plot: put
       inside, its own diamond glyph landed on top of a real camera at azimuth
       240 and the reader had two diamonds meaning different things */
    const c = makeChart(node, { width: 620, height: 380, margin: { top: 34, right: 152, bottom: 56, left: 60 } });
    const x = d3.scaleLinear().domain([-12, 372]).range([0, c.w]);
    const y = d3.scaleLinear().domain([-30, 84]).range([c.h, 0]);
    drawGrid(c.g, x, y, c.w, c.h, { xValues: [0, 90, 180, 270, 360], yValues: [-20, 0, 20, 40, 60, 80] });
    c.g.insert('rect', ':first-child')
      .attr('x', 0).attr('y', y(bandHi)).attr('width', c.w).attr('height', y(bandLo) - y(bandHi))
      .attr('fill', 'var(--primary)').attr('opacity', 0.09);
    drawAxes(c.g, x, y, c.w, c.h, { xValues: [0, 90, 180, 270, 360], yValues: [-20, 0, 20, 40, 60, 80] });
    axisLabels(c.g, c.w, c.h, { x: 'azimuth', y: 'elevation' });
    const inSub = new Set(meta.view_subsets[String(state.k)]);
    c.g.selectAll('circle.tr').data(data.cameras.train).join('circle').attr('class', 'tr')
      .attr('cx', (d) => x(d.az)).attr('cy', (d) => y(d.el))
      .attr('r', (d, i) => (inSub.has(i) ? 6 : 3.2))
      .attr('fill', (d, i) => (inSub.has(i) ? 'var(--primary)' : 'none'))
      .attr('stroke', 'var(--primary)').attr('stroke-width', 1.2)
      .attr('opacity', (d, i) => (inSub.has(i) ? 1 : 0.45));
    c.g.selectAll('circle.te').data(data.cameras.inside).join('circle').attr('class', 'te')
      .attr('cx', (d) => x(d.az)).attr('cy', (d) => y(d.el)).attr('r', 6)
      .attr('fill', 'none').attr('stroke', 'var(--anchor)').attr('stroke-width', 2.4);
    c.g.selectAll('path.ou').data(data.cameras.outside).join('path').attr('class', 'ou')
      .attr('d', d3.symbol(d3.symbolDiamond, 100))
      .attr('transform', (d) => `translate(${x(d.az)},${y(d.el)})`)
      .attr('fill', 'none').attr('stroke', 'var(--cosmos)').attr('stroke-width', 2.4);
    seriesLabel(c.g, 6, y(bandHi) - 8, `photographed band, ${bandLo}° to ${bandHi}°`,
      { colour: 'var(--primary)' });
    /* the glyphs in the legend are the glyphs on the chart, not dots standing
       in for them */
    const leg = c.g.append('g').attr('transform', `translate(${c.w + 10},10)`);
    const rows = [
      ['circle', 'var(--primary)', true, 'photographs used'],
      ['circle', 'var(--anchor)', false, 'held out, inside'],
      ['diamond', 'var(--cosmos)', false, 'held out, outside'],
    ];
    rows.forEach(([shape, col, fill], i) => {
      const gy = i * 19;
      if (shape === 'circle') {
        leg.append('circle').attr('cx', 6).attr('cy', gy).attr('r', 5.5)
          .attr('fill', fill ? col : 'none').attr('stroke', col).attr('stroke-width', 2);
      } else {
        leg.append('path').attr('d', d3.symbol(d3.symbolDiamond, 80))
          .attr('transform', `translate(6,${gy})`).attr('fill', 'none')
          .attr('stroke', col).attr('stroke-width', 2);
      }
      leg.append('text').attr('x', 18).attr('y', gy + 4).attr('font-size', 11)
        .attr('font-family', 'IBM Plex Mono, monospace').attr('fill', 'var(--squidink)')
        .text(rows[i][3]);
    });
    c.svg.append('text').attr('class', 'chart-title')
      .attr('x', c.width / 2).attr('y', 20).attr('text-anchor', 'middle')
      .text('where the cameras were');
  };

  const drawChart = () => {
    const node = document.querySelector('#views-chart');
    d3.select(node).selectAll('*').remove();
    const c = makeChart(node, { width: 600, height: 380, margin: { top: 34, right: 126, bottom: 56, left: 62 } });
    const xs = curve.map((r) => r.views);
    const x = d3.scalePoint().domain(xs.map(String)).range([0, c.w]).padding(0.5);
    /* short enough to fit the right margin at 10px, because a direct label
       that runs off the canvas is worse than a legend */
    const series = [
      { key: 'train', label: 'the photographs', colour: 'var(--stone)', get: (r) => r.train },
      { key: 'inside', label: 'held out, in band', colour: 'var(--anchor)', get: (r) => r.inside },
      { key: 'above', label: 'above the band', colour: 'var(--primary)', get: (r) => groupOf(r, above) },
      { key: 'below', label: 'below the band', colour: 'var(--cosmos)', get: (r) => groupOf(r, below) },
    ];
    const all = series.flatMap((s) => curve.map(s.get));
    const y = d3.scaleLinear().domain([Math.min(...all) - 1.5, Math.max(...all) + 1.5]).range([c.h, 0]).nice();
    drawGrid(c.g, x, y, c.w, c.h, { yTicks: 5 });
    drawAxes(c.g, x, y, c.w, c.h, { yTicks: 5 });
    axisLabels(c.g, c.w, c.h, { x: 'photographs used', y: 'PSNR' });
    series.forEach((s) => {
      const pts = curve.map((r) => [x(String(r.views)), y(s.get(r))]);
      c.g.append('path').datum(pts).attr('d', d3.line())
        .attr('fill', 'none').attr('stroke', s.colour).attr('stroke-width', 2.4);
      c.g.selectAll(`circle.p-${s.key}`).data(curve).join('circle').attr('class', `p-${s.key}`)
        .attr('cx', (r) => x(String(r.views))).attr('cy', (r) => y(s.get(r)))
        .attr('r', (r) => (r.views === state.k ? 6 : 3.5))
        .attr('fill', s.colour);
    });
    /* two of these series can land within a couple of pixels of each other, so
       the labels are pushed apart before they are drawn rather than after */
    const marks = spread(series.map((s) => ({
      y: y(s.get(curve[curve.length - 1])) + 4, label: s.label, colour: s.colour, yMax: c.h,
    })), 14);
    marks.forEach((m) => {
      seriesLabel(c.g, c.w + 6, m.y, m.label, { colour: m.colour });
    });
    c.svg.append('text').attr('class', 'chart-title')
      .attr('x', c.width / 2).attr('y', 20).attr('text-anchor', 'middle')
      .text('what more photographs buy, and where');
  };

  const drawSheet = () => {
    const cfg = data.sheets.views;
    const t = sheets.views.tiles;
    const host = document.querySelector('#views-sheet');
    host.innerHTML = '';
    [['a held-out camera inside the band', 0], ['a camera below the band, nobody stood here', cfg.cols]]
      .forEach(([title, off]) => {
        const h = document.createElement('div');
        h.className = 'cap';
        h.style.margin = '0.6rem 0 0.2rem';
        h.innerHTML = `<b>${title}</b>`;
        host.appendChild(h);
        const wrap = document.createElement('div');
        host.appendChild(wrap);
        const hl = cfg.labels.findIndex((L) => L === `${state.k} photos`);
        drawTiles(wrap, t.slice(off, off + cfg.cols), cfg.labels, cfg.tile,
          { max: 118, highlight: hl });
      });
  };

  const render = () => {
    drawRig();
    drawChart();
    drawSheet();
    const row = curve.find((r) => r.views === state.k);
    const first = curve[0];
    const last = curve[curve.length - 1];
    readout.innerHTML = `With <span class="bold">${state.k}</span> photographs the field holds `
      + `<span class="bold">${fmt.db(row.inside)}</span> on the held-out cameras inside the band and `
      + `<span class="bold">${fmt.db(groupOf(row, below))}</span> on the ones below it, a gap of `
      + `<span class="bold">${(row.inside - groupOf(row, below)).toFixed(2)} dB</span>. `
      + `Going from ${first.views} photographs to ${last.views} moves the inside number by `
      + `${(last.inside - first.inside).toFixed(2)} dB and the below-the-band number by `
      + `${(groupOf(last, below) - groupOf(first, below)).toFixed(2)} dB: `
      /* a tenth of a decibel between two gains is not a finding, so the margin
         has to be wide enough that the sentence means something */
      + ((last.inside - first.inside) - (groupOf(last, below) - groupOf(first, below)) > 0.4
        ? `more cameras in the same band do less for the part of the scene the band never saw.`
        : (groupOf(last, below) - groupOf(first, below)) - (last.inside - first.inside) > 0.4
          ? `here the cameras outside the band gained more than the ones inside, which the hull argument `
            + `does not predict.`
          : `on this scene the two move together, so the extra photographs are not buying the band `
            + `alone.`);
  };

  const paint = buttonRow(document.querySelector('#views-buttons'),
    curve.map((r) => ({ key: r.views, label: `${r.views} photographs` })),
    state.k, (k) => { state.k = k; paint(k); render(); });
  render();
}
