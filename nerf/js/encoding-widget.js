/* What the Fourier bands actually do, in closed form and in measurement.
 *
 * The left panel is arithmetic, not a picture: how far apart two points are in
 * the encoded space as a function of how far apart they are in the scene.
 * Without bands that distance is the distance itself, so a network whose
 * output has to change over a pixel has to have a huge slope; with L bands the
 * curve saturates within a fraction of a pixel, and after that the network can
 * say whatever it likes about the two points independently.
 *
 * The right panel is what that is worth once you actually train it, including
 * the control nobody runs: an encoding-free network with the same number of
 * weights, spent on width instead of on bands.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { drawTiles, seriesLabel, fmt } from './panels.js';

/* |gamma(x + d) - gamma(x)|, which is shift invariant and therefore a property
   of the encoding rather than of where you stand. The raw coordinate is kept
   in gamma, so L = 0 is the identity and the curve is the diagonal. */
export function encDistance(delta, L) {
  let s = delta * delta;
  for (let l = 0; l < L; l++) s += 2 * (1 - Math.cos(2 ** l * Math.PI * delta));
  return Math.sqrt(s);
}

/* The separation at which the encoding has first used up half its range: the
 * length scale below which the network cannot cheaply tell two points apart.
 *
 * Scan then bisect, and not bisect alone. The curve is a sum of cosines and
 * therefore not monotone: a plain bisection over [0, dmax] converges to
 * whichever crossing it happens to fall into, and with eight bands it returned
 * 0.97 scene units, the same answer as no bands at all, for a curve that
 * actually saturates two hundred times sooner. What is wanted is the FIRST
 * crossing, so the bracket has to be found by walking up from zero.
 */
export function halfSeparation(L, dmax = 2) {
  const target = 0.5 * encSaturation(L, dmax);
  const N = 4000 * Math.max(1, L);
  for (let k = 1; k <= N; k++) {
    const d = (k / N) * dmax;
    if (encDistance(d, L) >= target) {
      let lo = ((k - 1) / N) * dmax;
      let hi = d;
      for (let it = 0; it < 40; it++) {
        const mid = 0.5 * (lo + hi);
        if (encDistance(mid, L) < target) lo = mid; else hi = mid;
      }
      return 0.5 * (lo + hi);
    }
  }
  return dmax;
}

function encSaturation(L, dmax) {
  let m = 0;
  for (let k = 1; k <= 400; k++) m = Math.max(m, encDistance((k / 400) * dmax, L));
  return m;
}

export function initEncodingWidget(data, sheets) {
  const meta = data.meta;
  const ladder = data.nerf.encoding.ladder;
  const wide = data.nerf.encoding.wide;
  const half = (meta.aabb.hi[0] - meta.aabb.lo[0]) / 2;
  /* one pixel of a photograph, in scene units, at the radius the cameras sit */
  const pixel = (2 * meta.cam_radius * Math.tan((meta.fov_deg * Math.PI) / 360)) / meta.res;

  const slider = document.querySelector('#enc-l');
  const lval = document.querySelector('#enc-l-value');
  const readout = document.querySelector('#enc-readout');
  const state = { L: 6 };
  const LS = ladder.map((r) => r.L);
  slider.min = '0';
  slider.max = String(Math.max(...LS));

  const kernelNode = document.querySelector('#enc-kernel');
  const barNode = document.querySelector('#enc-bars');

  const drawKernel = () => {
    d3.select(kernelNode).selectAll('*').remove();
    const c = makeChart(kernelNode, { width: 560, height: 380, margin: { top: 34, right: 22, bottom: 56, left: 66 } });
    const dmaxWorld = 0.5;
    const dmin = pixel / 40;
    const x = d3.scaleLog().domain([dmin, dmaxWorld]).range([0, c.w]);
    const y = d3.scaleLinear().domain([0, 1.05]).range([c.h, 0]);
    /* ticks outside the domain do not disappear, they get drawn off the end of
       the axis: filter them against the domain rather than against a guess */
    const xv = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
      .filter((v) => v >= dmin && v <= dmaxWorld);
    drawGrid(c.g, x, y, c.w, c.h, { xValues: xv, yTicks: 5 });
    drawAxes(c.g, x, y, c.w, c.h, { xValues: xv, yTicks: 5, xFmt: d3.format('~g') });
    axisLabels(c.g, c.w, c.h, { x: 'separation in scene units', y: 'distance in the encoding' });

    const sample = d3.range(0, 401).map((k) => (k / 400) * (dmaxWorld / half));
    const draw = (L, active) => {
      const sat = encSaturation(L, 2);
      const pts = sample.filter((d) => d * half >= pixel / 40)
        .map((d) => [d * half, encDistance(d, L) / sat]);
      c.g.append('path').datum(pts)
        .attr('d', d3.line().x((p) => x(p[0])).y((p) => y(Math.min(1.02, p[1]))))
        .attr('fill', 'none')
        .attr('stroke', active ? 'var(--primary)' : 'var(--stone)')
        .attr('stroke-width', active ? 2.6 : 1.3);
      if (active) {
        const hs = halfSeparation(L) * half;
        const off = hs > dmaxWorld;
        const px = x(Math.min(Math.max(hs, dmin), dmaxWorld));
        c.g.append('circle').attr('cx', px).attr('cy', y(0.5)).attr('r', 4.5)
          .attr('fill', off ? 'none' : 'var(--primary)')
          .attr('stroke', 'var(--primary)').attr('stroke-width', 1.6);
        /* the marker slides the whole width of the chart as L changes, so the
           label changes which side of it it sits on; and with no bands at all
           the half point is past the end of the axis, which the label has to
           say rather than leave a filled dot sitting on the edge pretending */
        const right = px < c.w - 130;
        seriesLabel(c.g, px + (right ? 8 : -8), y(0.5) - 8,
          off ? `half past ${dmaxWorld}, off this axis`
            : `half at ${hs < 0.01 ? hs.toExponential(1) : hs.toFixed(3)}`,
          { anchor: right ? 'start' : 'end', colour: 'var(--primary)' });
      }
    };
    LS.forEach((L) => { if (L !== state.L) draw(L, false); });
    draw(state.L, true);

    c.g.append('line').attr('x1', x(pixel)).attr('x2', x(pixel)).attr('y1', 0).attr('y2', c.h)
      .attr('stroke', 'var(--anchor)').attr('stroke-dasharray', '4 3').attr('stroke-width', 1.4);
    seriesLabel(c.g, x(pixel) + 6, 14, 'one pixel', { colour: 'var(--anchor)' });
    c.svg.append('text').attr('class', 'chart-title')
      .attr('x', c.width / 2).attr('y', 20).attr('text-anchor', 'middle')
      .text('how quickly the encoding separates two points');
  };

  const drawBars = () => {
    d3.select(barNode).selectAll('*').remove();
    const c = makeChart(barNode, { width: 560, height: 380, margin: { top: 34, right: 22, bottom: 56, left: 62 } });
    const x = d3.scaleBand().domain(LS.map(String)).range([0, c.w]).padding(0.28);
    const lo = Math.min(wide.inside, ...ladder.map((r) => r.inside)) - 2;
    const hi = Math.max(...ladder.map((r) => r.inside)) + 1.5;
    const y = d3.scaleLinear().domain([lo, hi]).range([c.h, 0]).nice();
    drawGrid(c.g, x, y, c.w, c.h, { yTicks: 5 });
    drawAxes(c.g, x, y, c.w, c.h, { yTicks: 5 });
    axisLabels(c.g, c.w, c.h, { x: 'frequency bands L', y: 'held-out PSNR' });
    c.g.selectAll('rect.bar').data(ladder).join('rect').attr('class', 'bar')
      .attr('x', (d) => x(String(d.L))).attr('width', x.bandwidth())
      .attr('y', (d) => y(d.inside)).attr('height', (d) => c.h - y(d.inside))
      .attr('fill', (d) => (d.L === state.L ? 'var(--primary)' : 'var(--stone)'));
    ladder.forEach((d) => {
      seriesLabel(c.g, x(String(d.L)) + x.bandwidth() / 2, y(d.inside) - 7, d.inside.toFixed(1),
        { anchor: 'middle' });
    });
    c.g.append('line').attr('x1', 0).attr('x2', c.w)
      .attr('y1', y(wide.inside)).attr('y2', y(wide.inside))
      .attr('stroke', 'var(--cosmos)').attr('stroke-width', 2).attr('stroke-dasharray', '6 4');
    seriesLabel(c.g, c.w - 4, y(wide.inside) - 8,
      `no bands, width ${wide.W}: ${wide.inside.toFixed(1)}`,
      { anchor: 'end', colour: 'var(--cosmos)' });
    c.svg.append('text').attr('class', 'chart-title')
      .attr('x', c.width / 2).attr('y', 20).attr('text-anchor', 'middle')
      .text('what it is worth once trained');
  };

  const drawSheet = () => {
    const cfg = data.sheets.encoding;
    const idx = LS.indexOf(state.L);
    drawTiles(document.querySelector('#enc-sheet'), sheets.encoding.tiles,
      cfg.labels, cfg.tile, { max: 108, highlight: idx });
  };

  const render = () => {
    lval.textContent = String(state.L);
    drawKernel();
    drawBars();
    drawSheet();
    const row = ladder.find((r) => r.L === state.L);
    const best = ladder.reduce((a, b) => (b.inside > a.inside ? b : a));
    const zero = ladder.find((r) => r.L === 0);
    const hs = halfSeparation(state.L) * half;
    readout.innerHTML = `With <span class="bold">${state.L}</span> band${state.L === 1 ? '' : 's'} the `
      + `encoding is half of the way to telling two points apart after `
      + `<span class="bold">${hs < 0.01 ? hs.toExponential(2) : hs.toFixed(4)}</span> scene units, which is `
      + `<span class="bold">${(hs / pixel).toFixed(2)}</span> `
      + `${hs / pixel >= 1.5 ? 'pixels' : 'of a pixel'} at this camera distance. `
      + `Trained, it holds ${fmt.db(row.inside)} on the views it never saw, against `
      + `${fmt.db(zero.inside)} with no bands at all and ${fmt.db(wide.inside)} for the control that `
      + `spends the same ${fmt.n0(wide.params)} weights on width instead. `
      + `The best row here is L = ${best.L} at ${fmt.db(best.inside)}.`;
  };

  slider.addEventListener('input', () => { state.L = +slider.value; render(); });
  slider.value = '6';
  render();
}
