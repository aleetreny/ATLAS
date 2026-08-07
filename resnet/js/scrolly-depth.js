/* The degradation problem, told with the curves it actually produced.
 *
 * Every line here is a training run from depth.json, drawn on the training
 * loss because that is where the surprise lives: the deepest plain network
 * does not fail on data it has never seen, it fails on the data it was shown.
 */
import { scrolly } from '../../assets/js/scrolly.js';
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initScrollyDepth(data) {
  const byDepth = Object.fromEntries(data.ladder.map((r) => [r.depth, r]));
  const depths = data.ladder.map((r) => r.depth);
  const shallow = depths[0];
  const deepest = depths[depths.length - 1];
  /* The middle rung is the one where the plain family peaked, read from the
     measurement rather than picked: on this data it is not where the round
     number would have been, and the step headings say so. */
  const peakRow = data.ladder.reduce((a, b) => (b.plain.train_acc > a.plain.train_acc ? b : a));
  const middle = peakRow.depth;

  const words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
    'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
    'seventeen', 'eighteen', 'nineteen', 'twenty'];
  const spell = (n) => {
    if (n <= 20) return words[n];
    const tens = ['', '', 'twenty', 'thirty', 'forty', 'fifty'];
    const t = Math.floor(n / 10);
    const u = n % 10;
    return u ? `${tens[t]}-${words[u]}` : tens[t];
  };
  const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);

  const heads = document.querySelectorAll('#depth-scrolly .step h3');
  if (heads[0]) heads[0].textContent = `${cap(spell(shallow))} Layers`;
  if (heads[1]) heads[1].textContent = `${cap(spell(middle))} Layers`;
  if (heads[2]) heads[2].textContent = `${cap(spell(deepest))} Layers`;

  /* Both number-bearing paragraphs are written from the ladder, so a rerun
     that moves the peak cannot leave the prose behind. */
  const pctOf = (v) => `${(v * 100).toFixed(2)}%`;
  const midP = document.querySelector('#step-middle');
  if (midP) {
    midP.innerHTML = middle === shallow
      /* If the shallowest rung is already the best, there is no "deeper is
         better" step to narrate and the page must not pretend otherwise. */
      ? `On this ladder there is no rung where more layers helped: the shallowest network, `
        + `at ${shallow} layers, is already the best of its family on the training set with `
        + `<span class="bold">${pctOf(byDepth[shallow].plain.train_acc)}</span>.`
      : `More of the same layers, same everything else: `
        + `${pctOf(byDepth[shallow].plain.train_acc)} on the training set at ${shallow} layers becomes `
        + `<span class="bold">${pctOf(byDepth[middle].plain.train_acc)}</span> at ${middle}. `
        + `Going lower is the entire reason anybody started stacking. This is as far as it got.`;
  }
  const deepP = document.querySelector('#step-deep');
  if (deepP) {
    deepP.innerHTML = `Keep going and the curve moves the wrong way. At ${deepest} layers the same `
      + `family reaches <span class="bold">${pctOf(byDepth[deepest].plain.train_acc)}</span> on the data `
      + `it was trained on, against ${pctOf(byDepth[middle].plain.train_acc)} at ${middle}. `
      + `Not on data it had never seen: on the data it was shown. This is the measurement that made `
      + `residual connections necessary.`;
  }

  const W = 620;
  const H = 430;
  const margin = { top: 34, right: 150, bottom: 54, left: 70 };
  const { g, w, h } = makeChart('#depth-scrolly-chart', { width: W, height: H, margin });

  const epochs = byDepth[shallow].plain.history.length;
  const everything = data.ladder.flatMap((r) => [...r.plain.history, ...r.residual.history]);

  const x = d3.scaleLinear().domain([1, epochs]).range([0, w]);
  const y = d3.scaleLog()
    .domain([Math.min(...everything) * 0.85, Math.max(...everything) * 1.05])
    .range([h, 0]);

  drawGrid(g, x, y, w, h, { xTicks: epochs, yTicks: 5 });
  drawAxes(g, x, y, w, h, { xTicks: epochs, yTicks: 5, xFmt: d3.format('d'), yFmt: d3.format('.2f') });
  axisLabels(g, w, h, { x: 'epoch', y: 'training loss' });

  const plot = g.append('g');
  const legend = g.append('g');
  const notes = g.append('g');
  const line = d3.line().x((d, i) => x(i + 1)).y((d) => y(d));

  const PLAIN = '#232f3e';
  const GREY = '#a9b1bd';

  function show(entries, note = '') {
    plot.selectAll('*').remove();
    legend.selectAll('*').remove();
    notes.selectAll('*').remove();

    entries.forEach((e, i) => {
      const row = byDepth[e.depth][e.family];
      plot.append('path')
        .datum(row.history)
        .attr('fill', 'none')
        .attr('stroke', e.colour)
        .attr('stroke-width', e.width || 2.4)
        .attr('stroke-dasharray', e.dash || null)
        .attr('d', line);
      plot.append('circle')
        .attr('cx', x(epochs))
        .attr('cy', y(row.history[row.history.length - 1]))
        .attr('r', 3.5)
        .attr('fill', e.colour);
      legend.append('text')
        .attr('x', w + 10)
        .attr('y', 16 + i * 20)
        .attr('class', 'annotation')
        .style('font-size', '12px')
        .style('fill', e.colour)
        .text(e.label);
    });

    if (note) {
      /* In the top margin, not inside the plot: at these depths the first
         epoch's loss sits right where a caption at y=4 would land. */
      notes.append('text')
         .attr('x', 100)
        .attr('y', -14)
        .attr('class', 'annotation')
        .style('font-size', '12px')
        .style('fill', '#5b6572')
        .text(note);
    }
  }

  const fmt = (v) => `${(v * 100).toFixed(2)}%`;

  const steps = {
    0() {
      show([{ depth: shallow, family: 'plain', colour: PLAIN, label: `${shallow} layers` }],
        `training accuracy ${fmt(byDepth[shallow].plain.train_acc)}`);
    },
    1() {
      show([
        { depth: shallow, family: 'plain', colour: GREY, label: `${shallow} layers`, width: 1.8 },
        { depth: middle, family: 'plain', colour: PLAIN, label: `${middle} layers` },
      ], `${fmt(byDepth[shallow].plain.train_acc)} at ${shallow} layers, ${fmt(byDepth[middle].plain.train_acc)} at ${middle}`);
    },
    2() {
      show([
        { depth: shallow, family: 'plain', colour: GREY, label: `${shallow} layers`, width: 1.6 },
        { depth: middle, family: 'plain', colour: GREY, label: `${middle} layers`, width: 1.6 },
        { depth: deepest, family: 'plain', colour: '#b91c1c', label: `${deepest} layers` },
      ], `${deepest} layers: ${fmt(byDepth[deepest].plain.train_acc)} on its own training data`);
    },
    3() {
      const gd = byDepth[deepest].plain.grad_first_init;
      const gs = byDepth[shallow].plain.grad_first_init;
      show([
        { depth: middle, family: 'plain', colour: GREY, label: `${middle} layers`, width: 1.6 },
        { depth: deepest, family: 'plain', colour: '#b91c1c', label: `${deepest}, plain` },
      ], `gradient at layer one, at init: ${gs.toExponential(1)} at ${shallow} layers, `
        + `${gd.toExponential(1)} at ${deepest}`);
    },
    4() {
      show([
        { depth: middle, family: 'plain', colour: GREY, label: `${middle}, plain`, width: 1.6 },
        { depth: deepest, family: 'plain', colour: '#b91c1c', label: `${deepest}, plain` },
        { depth: deepest, family: 'residual', colour: 'var(--primary)', label: `${deepest}, shortcuts` },
      ], `same depth, same data, one addition: ${fmt(byDepth[deepest].residual.train_acc)}`);
    },
  };

  scrolly('#depth-scrolly .step', steps);
  steps[0]();
}
