/* Scrollytelling: 47 real stars, and the four that reverse the conclusion. */
import { scrolly } from '../../assets/js/scrolly.js';
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { ols, theilSen, ransac } from './mathkit.js';

const DUR = 700;

export function initScrollyStars(stars) {
  const pts = stars.points;
  const xs = pts.map((d) => d.x);
  const ys = pts.map((d) => d.y);
  const cleanIdx = pts.map((d, i) => i).filter((i) => !pts[i].giant);

  const fitAll = ols(xs, ys);
  const fitClean = ols(cleanIdx.map((i) => xs[i]), cleanIdx.map((i) => ys[i]));
  const ts = theilSen(xs, ys);
  // 0.6 is calibrated to the data: the main sequence scatters with sigma about
  // 0.4, so this admits 40 of the 43 honest stars and none of the giants.
  // A tighter threshold finds a narrow sub-band instead of the sequence.
  const rs = ransac(xs, ys, { threshold: 0.6, iterations: 400, seed: 11 });

  // runtime check against the scikit-learn fits shipped with the data
  const ref = stars.ref;
  const off = (a, b) => Math.abs(a - b);
  if (off(fitAll.slope, ref.ols.slope) > 0.01) console.warn(`ols: ${fitAll.slope} vs ${ref.ols.slope}`);
  if (off(fitClean.slope, ref.ols_giants_removed.slope) > 0.01) console.warn(`clean ols: ${fitClean.slope}`);
  if (off(ts.slope, ref.theilsen.slope) > 0.05) console.warn(`theil-sen: ${ts.slope} vs ${ref.theilsen.slope}`);

  const { g, plot, w, h } = makeChart('#stars-chart', {
    width: 600,
    height: 560,
    margin: { top: 46, right: 26, bottom: 62, left: 62 },
    clip: true,
  });

  const x = d3.scaleLinear().domain([3.35, 4.75]).range([0, w]);
  const y = d3.scaleLinear().domain([3.7, 6.5]).range([h, 0]);

  drawGrid(g, x, y, w, h);
  drawAxes(g, x, y, w, h);
  axisLabels(g, w, h, { x: 'log surface temperature', y: 'log light intensity' });

  const linesG = plot.append('g');
  const dotsG = g.append('g');
  const annG = g.append('g');

  const mkLine = (color, dash) =>
    linesG
      .append('line')
      .attr('stroke', color)
      .attr('stroke-width', 3.4)
      .attr('stroke-dasharray', dash || null)
      .attr('opacity', 0);

  const haloOls = linesG.append('line').attr('stroke', 'white').attr('stroke-width', 7).attr('opacity', 0);
  const lineOls = mkLine('var(--cosmos)');
  const lineClean = mkLine('var(--squidink)', '8 6');
  const lineRobust = mkLine('var(--primary)');

  const dots = dotsG
    .selectAll('circle')
    .data(pts)
    .join('circle')
    .attr('cx', (d) => x(d.x))
    .attr('cy', (d) => y(d.y))
    .attr('r', 5.5)
    .attr('fill', 'var(--squidink)')
    .attr('stroke', 'white')
    .attr('stroke-width', 1.3);

  const readout = annG
    .append('text')
    .attr('class', 'chart-annotation')
    .attr('x', w / 2)
    .attr('y', -20)
    .attr('text-anchor', 'middle');

  const t = () => d3.transition().duration(DUR).ease(d3.easeCubicOut);

  function setLine(el, halo, fit, visible) {
    const [x0, x1] = x.domain();
    for (const l of [halo, el].filter(Boolean)) {
      l.transition(t())
        .attr('x1', x(x0)).attr('y1', y(fit.intercept + fit.slope * x0))
        .attr('x2', x(x1)).attr('y2', y(fit.intercept + fit.slope * x1))
        .attr('opacity', visible ? 1 : 0);
    }
  }

  function setGiants({ highlight = false, fade = false } = {}) {
    dots
      .transition(t())
      .attr('fill', (d) => (d.giant && highlight ? 'var(--smile)' : 'var(--squidink)'))
      .attr('r', (d) => (d.giant && highlight ? 9 : 5.5))
      .attr('opacity', (d) => (d.giant && fade ? 0.18 : 1));
  }

  function setNote(text, place) {
    annG.selectAll('text.note').remove();
    if (!text) return;
    annG
      .append('text')
      .attr('class', 'chart-annotation note')
      .attr('x', place[0])
      .attr('y', place[1])
      .attr('text-anchor', 'middle')
      .attr('opacity', 0)
      .text(text)
      .transition(t())
      .attr('opacity', 1);
  }

  const handlers = {
    0: () => {
      setLine(lineOls, haloOls, fitAll, false);
      setLine(lineClean, null, fitClean, false);
      setLine(lineRobust, null, ts, false);
      setGiants({});
      setNote('', [0, 0]);
      readout.text('cyg ob1 · 47 stars');
    },
    1: () => {
      setLine(lineOls, haloOls, fitAll, true);
      setLine(lineClean, null, fitClean, false);
      setLine(lineRobust, null, ts, false);
      setGiants({});
      setNote('', [0, 0]);
      readout.text(`least squares · slope ${fitAll.slope.toFixed(2)}`);
    },
    2: () => {
      setLine(lineOls, haloOls, fitAll, true);
      setLine(lineClean, null, fitClean, false);
      setLine(lineRobust, null, ts, false);
      setGiants({ highlight: true });
      setNote('four red giants', [x(3.62), y(6.32)]);
      readout.text('four stars, upper left');
    },
    3: () => {
      setLine(lineOls, haloOls, fitAll, true);
      setLine(lineClean, null, fitClean, true);
      setLine(lineRobust, null, ts, false);
      setGiants({ highlight: true, fade: true });
      setNote('', [0, 0]);
      readout.text(`without them · slope ${fitClean.slope.toFixed(2)}`);
    },
    4: () => {
      setLine(lineOls, haloOls, fitAll, true);
      setLine(lineClean, null, fitClean, true);
      setLine(lineRobust, null, ts, false);
      setGiants({ highlight: true });
      setNote('far along x = long lever', [x(3.75), y(6.32)]);
      readout.text('leverage, not error size');
    },
    5: () => {
      setLine(lineOls, haloOls, fitAll, true);
      setLine(lineClean, null, fitClean, true);
      setLine(lineRobust, null, ts, true);
      setGiants({ highlight: true });
      setNote('', [0, 0]);
      readout.text(`theil-sen finds ${ts.slope.toFixed(2)} · ransac ${rs.slope.toFixed(2)}`);
    },
  };

  handlers[0]();
  scrolly('#stars-scrolly .step', handlers);
}
