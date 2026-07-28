/* The claim of the article, computed in front of the reader.
 *
 * Two views of one layer. The first draws the convolution kernel, which is the
 * layer's entire memory written out: how much an input from t steps ago still
 * contributes. The second draws both outputs on the same axes, the step-by-step
 * recurrence and the all-at-once convolution, and reports the worst
 * disagreement between them.
 *
 * The slider changes how slowly the state forgets, which is the one thing that
 * makes this identity harder to satisfy: slow dynamics accumulate hundreds of
 * terms, so the error grows with it. Watching that number stay in the region of
 * 1e-14 across the whole range is the point.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { seriesLabel, legend } from '../../assets/js/textdraw.js';
import { recurrent, kernel, convolve, dualityError, reachOf } from './ssmkit.js';

const VIEWS = [
  { key: 'kernel', label: 'the memory, written out' },
  { key: 'both', label: 'both computations, one answer' },
];
const DECAYS = [0.5, 0.7, 0.85, 0.93, 0.97, 0.99, 0.995, 0.999];

/**
 * @param demo the DECODED system from `loadDemo`, not `data.demo`.
 *
 * Worth stating in a signature, because the raw object carries `b`, `c` and `x`
 * as base64 strings and every one of them indexes without complaining: `b[i]`
 * on a string is a character, a character times a number is NaN, and NaN plots
 * as an empty chart with correct axes. The same shape of mistake left an
 * attention matrix invisible earlier in this module.
 */
export function initDualityWidget(demo) {
  const D = demo;
  if (typeof D.b === 'string' || typeof D.c === 'string' || typeof D.x === 'string') {
    throw new Error('the duality widget was handed the raw JSON instead of the decoded system');
  }
  let view = 'kernel';
  let di = 4;

  const chart = makeChart('#duality-chart', {
    width: 640, height: 380, margin: { top: 44, right: 26, bottom: 56, left: 72 },
  });
  const { g, w, h } = chart;
  const readout = d3.select('#duality-readout');

  d3.select('#duality-views').selectAll('button').data(VIEWS).join('button')
    .attr('class', (d) => `atlas-btn${d.key === view ? '' : ' ghost'}`)
    .text((d) => d.label)
    .on('click', (event, d) => {
      view = d.key;
      d3.select('#duality-views').selectAll('button')
        .attr('class', (q) => `atlas-btn${q.key === view ? '' : ' ghost'}`);
      render();
    });

  /* The slider indexes the measured list rather than a range of numbers, which
   * is the rule this site learned the hard way: a control that walks off its
   * own table dies silently in the middle of its travel. */
  const slider = d3.select('#duality-decay')
    .attr('min', 0).attr('max', DECAYS.length - 1).attr('step', 1).property('value', di)
    .on('input', () => { di = +slider.property('value'); render(); });

  /* One state per decay setting, built from the shipped parameters by rescaling
   * the multipliers, so every setting is the same layer with one thing changed. */
  function paramsFor(decay) {
    const n = D.n;
    const a = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      /* spread the multipliers below the chosen decay, the way a structured
       * initialisation does, rather than making them all equal */
      a[i] = decay ** (1 + (i / Math.max(1, n - 1)));
    }
    return { a, b: D.b, c: D.c, x: D.x };
  }

  function render() {
    const decay = DECAYS[di];
    const { a, b, c, x } = paramsFor(decay);
    d3.select('#duality-decay-label').text(decay.toFixed(3));
    const k = kernel(a, b, c, D.L);
    const err = dualityError(a, b, c, x);

    g.selectAll('*').remove();
    if (view === 'kernel') {
      const k0 = Math.abs(k[0]) || 1;
      const rel = Array.from(k).map((v, t) => ({ t, v: Math.abs(v) / k0 }));
      const lo = Math.max(1e-8, Math.min(...rel.map((p) => p.v).filter((v) => v > 0)));
      const x0 = d3.scaleLinear().domain([0, D.L]).range([0, w]);
      const y0 = d3.scaleLog().domain([Math.max(lo, 1e-6), 1.4]).range([h, 0]);
      g.append('g').attr('class', 'grid')
        .call((s) => drawGrid(s, x0, y0, w, h, { xTicks: 6, yTicks: 5 }));
      const axes = g.append('g');
      drawAxes(axes, x0, y0, w, h, { xTicks: 6, yTicks: 5,
        yFmt: (v) => (v >= 0.01 ? String(v) : v.toExponential(0)) });
      axisLabels(axes, w, h, {
        x: 'steps ago', y: 'how much that input still contributes',
      });
      g.append('path').datum(rel.filter((p) => p.v > 0))
        .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2.2)
        .attr('d', d3.line().x((p) => x0(p.t)).y((p) => y0(Math.max(p.v, y0.domain()[0]))));
      const hund = reachOf(k, 0.01);
      if (hund !== null) {
        g.append('line').attr('x1', x0(hund)).attr('x2', x0(hund))
          .attr('y1', 0).attr('y2', h)
          .attr('stroke', 'var(--cosmos)').attr('stroke-width', 1.4)
          .attr('stroke-dasharray', '4 3');
        const t = seriesLabel(g, x0(hund) + 6, 16, `down to a hundredth at ${hund}`,
          { anchor: 'start', size: 10.5, colour: 'var(--cosmos)' });
        if (x0(hund) + 6 + t.node().getComputedTextLength() > w) {
          t.attr('x', x0(hund) - 6).attr('text-anchor', 'end');
        }
      }
      legend(g, [{ label: 'the convolution kernel', colour: 'var(--primary)', shape: 'line' }],
        { x0: 0, y0: -22, gap: 180, size: 10.5 });
    } else {
      const n = Math.min(96, D.L);
      const x0 = d3.scaleLinear().domain([0, n - 1]).range([0, w]);
      const all = [...Array.from(err.y).slice(0, n), ...Array.from(err.yc).slice(0, n)];
      const y0 = d3.scaleLinear().domain([Math.min(...all) * 1.1, Math.max(...all) * 1.1])
        .range([h, 0]);
      g.append('g').attr('class', 'grid')
        .call((s) => drawGrid(s, x0, y0, w, h, { xTicks: 6, yTicks: 5 }));
      const axes = g.append('g');
      drawAxes(axes, x0, y0, w, h, { xTicks: 6, yTicks: 5, yFmt: (v) => v.toFixed(1) });
      axisLabels(axes, w, h, { x: 'position in the sequence', y: 'output' });
      const line = (arr, colour, width, dash) => {
        g.append('path').datum(Array.from(arr).slice(0, n).map((v, t) => ({ t, v })))
          .attr('fill', 'none').attr('stroke', colour).attr('stroke-width', width)
          .attr('stroke-dasharray', dash)
          .attr('d', d3.line().x((p) => x0(p.t)).y((p) => y0(p.v)));
      };
      line(err.y, 'var(--primary)', 3.2, null);
      line(err.yc, 'var(--cosmos)', 1.6, '5 4');
      legend(g, [
        { label: 'step by step', colour: 'var(--primary)', shape: 'line' },
        { label: 'all at once', colour: 'var(--cosmos)', shape: 'dash' },
      ], { x0: 0, y0: -22, gap: 170, size: 10.5 });
    }

    const hund = reachOf(k, 0.01);
    readout.html(
      `<div>With the state forgetting at <span class="bold">${decay.toFixed(3)}</span> per step, `
      + `an input still contributes a hundredth of its initial weight `
      + `<span class="bold">${hund === null ? `more than ${D.L}` : hund}</span> steps later, `
      + `from a state of only <span class="value">${D.n}</span> numbers.</div>`
      + `<div>The two computations of that same layer, over `
      + `<span class="value">${D.L}</span> positions, disagree by at most `
      + `<span class="bold">${err.worst.toExponential(2)}</span> relative to the size of the `
      + `output. That is the accumulated rounding of a few hundred additions and nothing else: `
      + `they are not similar, they are the same function written twice.</div>`
      + `<div>Which is the point. The right-hand form can be evaluated for every position at `
      + `once, so training parallelises over the sequence the way attention does. The left-hand `
      + `form needs ${D.n} numbers of state and one step of work per new token, so generation `
      + `costs the same at position 10 and position 10,000. No other family in this branch gets `
      + `to pick.</div>`
    );
  }

  render();
  return { render };
}
