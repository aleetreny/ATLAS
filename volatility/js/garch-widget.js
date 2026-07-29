/* The variance recursion, run in the page on all of the real returns.
 *
 * Three numbers produce a path nobody ever observes, and the only thing that
 * scores it is a likelihood, so the likelihood is computed here too: the
 * readout is the reader's negative log likelihood against the one the fitting
 * library reached. The fit button snaps to the nearest position the sliders
 * can express, which is not quite the maximum, and the gap between those two
 * is reported rather than hidden, because a reference the control cannot reach
 * is a reference that makes the widget look broken.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';
import { garchFilter, garchNLL } from '../../assets/js/series.js';

/* omega is on a log scale because the two settings worth reaching are two
   orders of magnitude apart: the fitted value is around 0.017 with a long
   memory, and a model with no memory at all needs it to be the whole sample
   variance, around 1.45. A linear slider covering both has no resolution
   anywhere near the first. */
const OMEGA = (i) => 0.01 * 10 ** (i / 40);   // 0..100 spans 0.01 to 3.16
const OMEGA_I = (v) => Math.round(40 * Math.log10(v / 0.01));
const ALPHA = (i) => i * 0.005;               // 0..60
const BETA = (i) => i * 0.01;                 // 0..99

export function initGarchWidget(data) {
  const G = data.garch;
  const y = data.returns.y.map((v) => v - G.mean);
  const labels = data.returns.labels;
  const readout = document.querySelector('#garch-readout');
  const sliders = {
    omega: document.querySelector('#omega-slider'),
    alpha: document.querySelector('#alpha-slider'),
    beta: document.querySelector('#beta-slider'),
  };
  sliders.omega.min = 0; sliders.omega.max = 100; sliders.omega.value = 40;
  sliders.alpha.min = 0; sliders.alpha.max = 60; sliders.alpha.value = 8;
  sliders.beta.min = 0; sliders.beta.max = 99; sliders.beta.value = 55;

  const chart = makeChart('#garch-chart', {
    width: 640, height: 380, margin: { top: 44, right: 20, bottom: 48, left: 62 },
  });
  const { g, w, h, margin } = chart;
  const step = Math.max(1, Math.round(y.length / 1400));
  const idx = d3.range(0, y.length, step);
  const x = d3.scaleLinear().domain([0, y.length - 1]).range([0, w]);
  const yv = d3.scaleLinear().domain([0, 90]).range([h, 0]);
  const years = ['2000', '2004', '2008', '2012', '2016']
    .map((p) => labels.findIndex((l) => l.startsWith(p))).filter((i) => i >= 0);
  drawGrid(g, x, yv, w, h, { xValues: years, yTicks: 5 });
  drawAxes(g, x, yv, w, h, { xValues: years, yTicks: 5, xFmt: (i) => labels[i].slice(0, 4) });
  axisLabels(g, w, h, { y: 'volatility, annualised percent', leftBudget: margin.left });

  /* the realised size of each day, as a faint background: this is all anyone
     ever sees of the quantity the line is a model of */
  g.selectAll('line.obs').data(idx).join('line').attr('class', 'obs')
    .attr('x1', (i) => x(i)).attr('x2', (i) => x(i))
    .attr('y1', yv(0)).attr('y2', (i) => yv(Math.min(90, Math.abs(y[i]) * Math.sqrt(252))))
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 0.7).attr('opacity', 0.25);
  const fitted = g.append('path').attr('fill', 'none').attr('stroke', 'var(--anchor)')
    .attr('stroke-width', 1.6).attr('opacity', 0.55).attr('stroke-dasharray', '5 4');
  const path = g.append('path').attr('fill', 'none').attr('stroke', 'var(--primary)')
    .attr('stroke-width', 1.8);
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -24).attr('text-anchor', 'middle').style('font-size', '12px');
  g.append('text').attr('class', 'chart-note').attr('x', 2).attr('y', -8)
    .style('font-size', '11px').attr('fill', 'var(--anchor)')
    .text('dashed: the fitted model. grey: the size of each day');

  const line = d3.line().x((i) => x(i)).y((v) => v);
  const draw = (sel, vol) => sel.datum(idx)
    .attr('d', d3.line().x((i) => x(i))
      .y((i) => yv(Math.min(90, vol[i] * Math.sqrt(252))))(idx));

  const mlVol = G.vol;
  draw(fitted, mlVol);

  function current() {
    return {
      omega: OMEGA(+sliders.omega.value),
      alpha: ALPHA(+sliders.alpha.value),
      beta: BETA(+sliders.beta.value),
    };
  }

  function render() {
    const { omega, alpha, beta } = current();
    const pers = alpha + beta;
    const start = omega + pers * G.backcast;
    const v = garchFilter(y, omega, alpha, beta, start);
    const nll = garchNLL(y, omega, alpha, beta, start);
    draw(path, v.map(Math.sqrt));
    document.querySelector('#omega-value').textContent = omega.toFixed(3);
    document.querySelector('#alpha-value').textContent = alpha.toFixed(3);
    document.querySelector('#beta-value').textContent = beta.toFixed(3);
    wrapLabel(title, `omega ${omega.toFixed(3)}, alpha ${alpha.toFixed(3)}, `
      + `beta ${beta.toFixed(3)}: persistence ${pers.toFixed(4)}`, w - 8);

    const half = pers < 1 ? Math.log(0.5) / Math.log(pers) : Infinity;
    const uncond = pers < 1 ? omega / (1 - pers) : null;
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th></th><th>yours</th><th>the fitted model</th></tr>
        <tr><td>negative log likelihood, all
                ${data.meta.ret_n.toLocaleString('en-US')} days</td>
            <td><span class="value">${nll.toFixed(1)}</span></td>
            <td>${G.nll}</td></tr>
        <tr><td>persistence, alpha + beta</td>
            <td><span class="value">${pers.toFixed(4)}</span></td>
            <td>${G.persistence}</td></tr>
        <tr><td>half life of a shock, trading days</td>
            <td>${pers < 1 ? half.toFixed(1) : 'never decays'}</td>
            <td>${G.half_life}</td></tr>
        <tr><td>long run volatility, annualised</td>
            <td>${uncond === null ? 'none: it diverges'
    : `${(Math.sqrt(uncond * 252)).toFixed(2)}%`}</td>
            <td>${G.uncond_vol_annual}%</td></tr>
      </table>
      <div class="gen-note">
        ${nll - G.nll < 1
    ? `That is within one nat of the best fit, which is as close as these
           sliders can be set: the maximum sits at omega ${G.omega}, alpha
           ${G.alpha}, beta ${G.beta}, and the nearest positions the controls
           can express cost ${(nll - G.nll).toFixed(2)}.`
    : `The fitted model is ${(nll - G.nll).toFixed(1)} nats better. The fit
           button moves the three sliders to the closest they can get to it.`}
        ${pers >= 1
    ? ` And this setting has no long run volatility at all: with alpha + beta at
           or above one, a shock never fades and the forecast of the variance
           runs away to infinity. The fitted value is ${G.persistence}, which is
           close enough to that edge to be worth staring at: a shock is half
           forgotten after ${G.half_life} trading days, about
           ${(G.half_life / 21).toFixed(1)} months.`
    : ` A shock is half forgotten after ${half.toFixed(0)} trading days here,
           against ${G.half_life} for the fitted model.`}
      </div>`;
  }

  d3.select('#garch-controls').selectAll('button').data([
    ['fit it', 'fit'], ['no memory at all', 'flat'], ['a shock that never fades', 'walk'],
  ]).join('button')
    .attr('class', 'atlas-btn ghost')
    .text((d) => d[0])
    .on('click', (evt, d) => {
      if (d[1] === 'fit') {
        sliders.omega.value = String(OMEGA_I(G.omega));
        sliders.alpha.value = String(Math.round(G.alpha / 0.005));
        sliders.beta.value = String(Math.round(G.beta / 0.01));
      } else if (d[1] === 'flat') {
        sliders.omega.value = String(OMEGA_I(G.sample_var));
        sliders.alpha.value = '0';
        sliders.beta.value = '0';
      } else {
        sliders.omega.value = '0';
        sliders.alpha.value = '10';
        sliders.beta.value = '99';
      }
      render();
    });

  Object.values(sliders).forEach((s) => s.addEventListener('input', render));
  render();
  return { render, current };
}
