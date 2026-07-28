/* Two unbiased estimators of the same derivative, run here, side by side.
 *
 * The version on this page is small enough to have a closed form, which is the
 * only reason it is worth running live: with q = N(mu, s squared) and
 * f(z) = -(z - c) squared, both estimators average to -2(mu - c) and their
 * variances are 4 s squared and 15 s squared + 14 a squared + a to the fourth
 * over s squared. So the spread the reader watches can be checked against
 * algebra rather than against another simulation, and the generator exports
 * both the algebra and its own Monte Carlo check of it.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';
import { lcg } from '../../assets/js/halfweights.js';

const SIZES = [1, 2, 4, 16, 64, 256];
const REPEATS = 260;

export function initTrickWidget(data) {
  const D = data.estimators.demo;
  const rows = data.estimators.rows;
  const chart = makeChart('#trick-chart', {
    width: 640, height: 330, margin: { top: 54, right: 22, bottom: 56, left: 30 },
  });
  const { g, w, h } = chart;
  const slider = document.querySelector('#trick-slider');
  const valueTag = document.querySelector('#trick-value');
  const readout = document.querySelector('#trick-readout');
  slider.min = 0;
  slider.max = SIZES.length - 1;
  slider.value = 0;

  const layer = g.append('g');
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -34).attr('text-anchor', 'middle')
    .style('font-size', '12px');

  /* Both estimators, from the same draws, exactly as they are written down. */
  function estimates(n, seed) {
    const rng = lcg(seed);
    const path = [];
    const score = [];
    for (let r = 0; r < REPEATS; r++) {
      let sp = 0;
      let ss = 0;
      for (let i = 0; i < n; i++) {
        const eps = rng.normal();
        const z = D.mu + D.sigma * eps;
        sp += -2 * (z - D.c);
        ss += -((z - D.c) ** 2) * (z - D.mu) / (D.sigma * D.sigma);
      }
      path.push(sp / n);
      score.push(ss / n);
    }
    return { path, score };
  }

  function sd(v) {
    const m = v.reduce((a, b) => a + b, 0) / v.length;
    return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / (v.length - 1));
  }

  function mean(v) {
    return v.reduce((a, b) => a + b, 0) / v.length;
  }

  function render() {
    const n = SIZES[+slider.value];
    valueTag.textContent = n === 1 ? '1 sample' : `${n} samples`;
    const { path, score } = estimates(n, 20250728 + n);

    /* The axis is set by the wider cloud, always, so that the tighter one is
     * seen to be tighter instead of being rescaled into looking the same. */
    const lo = Math.min(...score, ...path, D.true_gradient);
    const hi = Math.max(...score, ...path, D.true_gradient);
    const pad = (hi - lo) * 0.06 + 1e-9;
    const x = d3.scaleLinear().domain([lo - pad, hi + pad]).range([0, w]);
    const y = d3.scalePoint().domain(['pathwise', 'score function']).range([70, h - 60]);

    drawGrid(g, x, y, w, h, { xTicks: 7, yValues: [] });
    drawAxes(g, x, y, w, h, { xTicks: 7, yValues: [] });
    axisLabels(g, w, h, { x: 'one estimate of the same derivative', y: '' });

    layer.selectAll('*').remove();
    layer.append('line').attr('x1', x(D.true_gradient)).attr('x2', x(D.true_gradient))
      .attr('y1', 24).attr('y2', h - 10)
      .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.6)
      .attr('stroke-dasharray', '6 4');
    layer.append('text').attr('class', 'chart-note')
      .attr('x', x(D.true_gradient) > w * 0.6 ? x(D.true_gradient) - 8 : x(D.true_gradient) + 8)
      .attr('y', 16).attr('text-anchor', x(D.true_gradient) > w * 0.6 ? 'end' : 'start')
      .style('font-size', '11.5px')
      .text(`the answer, ${D.true_gradient}`);

    const rng = lcg(7);
    [['pathwise', path, 'var(--primary)'], ['score function', score, 'var(--anchor)']]
      .forEach(([name, vals, colour]) => {
        layer.selectAll(`circle.${name.replace(' ', '-')}`).data(vals).join('circle')
          .attr('class', name.replace(' ', '-'))
          .attr('cx', (d) => x(d))
          .attr('cy', () => y(name) + (rng.uniform() - 0.5) * 46)
          .attr('r', 2.6).attr('fill', colour).attr('opacity', 0.55);
        layer.append('text').attr('class', 'chart-note')
          .attr('x', 4).attr('y', y(name) - 34)
          .style('font-size', '11.5px')
          .attr('fill', colour).text(name);
      });

    wrapLabel(title, `${REPEATS} independent estimates from each rule, every one of them `
      + `averaging ${n} draw${n === 1 ? '' : 's'}`, w - 8);

    const sp = sd(path);
    const ss = sd(score);
    const k2 = rows.find((r) => r.k === 2);
    readout.innerHTML = `
      <table class="gen-table">
        <tr>
          <th>rule</th><th>average of ${REPEATS} estimates</th><th>spread</th>
          <th>predicted spread</th><th>what it costs</th>
        </tr>
        <tr>
          <td>pathwise</td>
          <td>${mean(path).toFixed(3)}</td>
          <td><span class="value">${sp.toFixed(3)}</span></td>
          <td>${Math.sqrt(D.var_pathwise / n).toFixed(3)}</td>
          <td>one backward pass through the sample</td>
        </tr>
        <tr>
          <td>score function</td>
          <td>${mean(score).toFixed(3)}</td>
          <td><span class="value">${ss.toFixed(3)}</span></td>
          <td>${Math.sqrt(D.var_score / n).toFixed(3)}</td>
          <td>nothing: it never differentiates the objective</td>
        </tr>
      </table>
      <div class="gen-note">
        Both clouds are centred on the same answer, because both rules are unbiased: over
        ${D.draws.toLocaleString('en-US')} draws the generator measured
        ${D.measured_mean_pathwise} and ${D.measured_mean_score} against the exact
        ${D.true_gradient}. What separates them is the width, and the width is not an
        accident of this simulation: the two variances are ${D.var_pathwise} and
        ${D.var_score} in closed form, a factor of <span class="value">${D.ratio}</span>,
        and averaging ${n} draw${n === 1 ? '' : 's'} divides both by ${n} and leaves the
        factor exactly where it was. That is the point: you cannot buy your way out of it
        with more samples, you have to change the rule.
        In the real network above, on the encoder's ${k2.params.toLocaleString('en-US')}
        weights, the same measurement gives <span class="value">${k2.ratio.toLocaleString('en-US')}</span>
        at ${k2.k} latent numbers.
      </div>`;
  }

  slider.addEventListener('input', render);
  render();
  return { render };
}
