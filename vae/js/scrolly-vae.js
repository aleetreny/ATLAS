/* The mechanism, one change at a time, on the plane the rest of the page uses.
 *
 * Everything drawn here is in the file: the points are the codes the two
 * encoders produced for the same 2,000 digits, the ellipses are the widths the
 * variational encoder returned for each of them, and the crosses are draws
 * from the prior made with the same generator the python side wrote, so the
 * picture is not an illustration of the idea, it is the idea's output.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';
import { scrolly } from '../../assets/js/scrolly.js';
import { toRows } from '../../assets/js/embed.js';
import { unitDraws } from './vaekit.js';

const DIGIT = d3.scaleOrdinal(d3.schemeCategory10).domain(d3.range(10));

export function initScrollyVae(data) {
  const N = data.net;
  const AEZ = toRows(N.ae_latent);
  const MU = toRows(N.mu);
  const LV = toRows(N.lv);
  const chart = makeChart('#vae-chart', {
    width: 560, height: 470, margin: { top: 54, right: 26, bottom: 54, left: 58 },
  });
  const { g, w, h } = chart;

  let lim = 0;
  AEZ.concat(MU).forEach((p) => { lim = Math.max(lim, Math.abs(p[0]), Math.abs(p[1])); });
  lim = Math.min(lim, 6) * 1.04;
  const x = d3.scaleLinear().domain([-lim, lim]).range([0, w]);
  const y = d3.scaleLinear().domain([-lim, lim]).range([h, 0]);

  const dots = g.append('g');
  const ell = g.append('g');
  const prior = g.append('g');
  const marks = g.append('g');
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -32).attr('text-anchor', 'middle')
    .style('font-size', '12px');

  const draws = unitDraws(2, 40, 8191);
  const perRowKl = MU.map((m, i) => {
    let s = 0;
    for (let j = 0; j < m.length; j++) {
      s += 0.5 * (m[j] * m[j] + Math.exp(LV[i][j]) - 1 - LV[i][j]);
    }
    return s;
  });
  const klMax = d3.quantile([...perRowKl].sort(d3.ascending), 0.98);
  const heat = d3.scaleLinear().domain([0, klMax || 1]).range(['#d9cfc9', '#7c2d12'])
    .clamp(true);
  const SHOWN = 220;                 // ellipses, thinned so the picture stays readable

  function base() {
    drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
    drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
    axisLabels(g, w, h, { x: 'first number', y: 'second number' });
  }

  function render(step) {
    base();
    const usingAe = step === 0;
    const Z = usingAe ? AEZ : MU;
    dots.selectAll('circle').data(Z).join('circle')
      .attr('cx', (d) => x(d[0])).attr('cy', (d) => y(d[1]))
      .attr('r', 2)
      .attr('fill', (d, i) => (step === 4 ? heat(perRowKl[i]) : DIGIT(N.labels[i])))
      .attr('opacity', step === 1 ? 0.28 : 0.62);

    const wantEll = step === 1 || step === 2;
    ell.selectAll('ellipse')
      .data(wantEll ? d3.range(0, MU.length, Math.ceil(MU.length / SHOWN)) : [])
      .join('ellipse')
      .attr('cx', (i) => x(MU[i][0])).attr('cy', (i) => y(MU[i][1]))
      .attr('rx', (i) => Math.abs(x(Math.exp(0.5 * LV[i][0])) - x(0)))
      .attr('ry', (i) => Math.abs(y(Math.exp(0.5 * LV[i][1])) - y(0)))
      .attr('fill', (i) => DIGIT(N.labels[i])).attr('opacity', 0.16)
      .attr('stroke', (i) => DIGIT(N.labels[i])).attr('stroke-opacity', 0.5);

    prior.selectAll('circle').data(step >= 2 && step <= 3 ? [1, 2] : []).join('circle')
      .attr('cx', x(0)).attr('cy', y(0))
      .attr('r', (s) => Math.abs(x(s) - x(0)))
      .attr('fill', 'none').attr('stroke', 'var(--squidink)')
      .attr('stroke-dasharray', '5 4').attr('stroke-width', 1.5);

    marks.selectAll('path').data(step === 3 ? draws : []).join('path')
      .attr('d', d3.symbol().type(d3.symbolCross).size(64))
      .attr('transform', (d) => `translate(${x(d[0])},${y(d[1])}) rotate(45)`)
      .attr('fill', 'var(--primary)').attr('opacity', 0.9);

    const labels = [
      'the codes the autoencoders article published: one point per digit',
      'the same digits, now a cloud each, drawn at one standard deviation',
      'and every cloud pulled towards the same standard gaussian',
      `${draws.length} draws from that gaussian, which is what sampling is`,
      'what each digit pays in nats for sitting where it sits',
    ];
    wrapLabel(title, labels[step], w - 8);
  }

  render(0);
  const handlers = {};
  for (let i = 0; i < 5; i++) handlers[i] = () => render(i);
  const ctl = scrolly('#vae-scrolly .step', handlers);
  return { render, goTo: ctl.goTo };
}
