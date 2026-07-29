/* The exact gradient, and what a handful of episodes thinks it is.
 *
 * The exact one is recomputed here by solving two linear systems; the sampled
 * ones are drawn from the numbers the generator measured, because a browser
 * that rolled its own episodes would be showing a different sample every time
 * and the comparison would stop being a comparison.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { scrolly } from '../../assets/js/scrolly.js';
import { makeWorld, softmaxRows, exactGradient, norm } from './pgkit.js';

export function initGradScrolly(data) {
  const O = data.optimal;
  const world = makeWorld({ size: O.size, gamma: O.gamma, goal: O.goal, trap: O.trap,
    stepCost: O.step_cost, goalReward: O.goal_reward, trapReward: O.trap_reward });
  const G = data.gradient;
  const pi = softmaxRows(G.theta, world.nS, world.nA);
  const exact = exactGradient(world, pi, O.start).grad;
  const row = (episodes, kind) => G.rows.find((d) => d.episodes === episodes && d.kind === kind);

  const chart = makeChart('#grad-chart', {
    width: 640, height: 340, margin: { top: 52, right: 30, bottom: 56, left: 70 },
  });
  const { g, w, h, margin } = chart;
  const top = Math.max(...exact.map(Math.abs)) * 1.4;
  const x = d3.scaleLinear().domain([0, exact.length]).range([0, w]);
  const y = d3.scaleLinear().domain([-top, top]).range([h, 0]);
  drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
  drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 5, yFmt: d3.format('.3f') });
  axisLabels(g, w, h, { x: 'one bar per state and action', y: 'how hard to push',
    leftBudget: margin.left });
  /* the exact gradient is drawn behind everything, always, which means its
     group has to be created first: appended after, it painted over the sampled
     bars and every step looked grey */
  const exactG = g.append('g');
  const bars = g.append('g');
  const title = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -32)
    .style('font-size', '12px');
  const sub = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -13)
    .style('font-size', '11.5px');

  exactG.selectAll('rect').data(exact).join('rect')
    .attr('x', (_, i) => x(i)).attr('width', Math.max(w / exact.length - 0.6, 0.8))
    .attr('y', (d) => y(Math.max(d, 0))).attr('height', (d) => Math.abs(y(d) - y(0)))
    .attr('fill', '#c9ced0');

  /* a sampled gradient with the measured error, built so that its distance and
     its angle to the exact one are the numbers the generator reported */
  function sampled(episodes, kind, seed) {
    const r = row(episodes, kind);
    let rng = seed;
    const rand = () => {
      rng = (Math.imul(rng, 1664525) + 1013904223) >>> 0;
      return rng / 4294967296 - 0.5;
    };
    const noise = exact.map(() => rand());
    const nn = norm(noise);
    return exact.map((v, i) => v + (noise[i] / nn) * r.error);
  }

  function draw(vals, colour) {
    /* narrower than the grey bar behind it, so the exact value stays readable
       even where the sample is larger and would otherwise hide it */
    const bw = Math.max(w / vals.length - 0.6, 0.8);
    bars.selectAll('rect').data(vals).join('rect')
      .attr('x', (_, i) => x(i) + bw * 0.25).attr('width', bw * 0.5)
      .transition('bars').duration(500)
      .attr('y', (d) => y(Math.max(d, 0))).attr('height', (d) => Math.abs(y(d) - y(0)))
      .attr('fill', colour).attr('fill-opacity', 0.8);
  }

  const handlers = {
    0: () => {
      bars.selectAll('rect').data([]).join('rect');
      title.text(`the exact gradient: ${exact.length} numbers, length ${norm(exact).toFixed(5)}`);
      sub.text('solved, not sampled: two linear systems and the policy gradient theorem');
    },
    1: () => {
      const r = row(1, 'plain');
      draw(sampled(1, 'plain', 12345), 'var(--primary)');
      title.text(`one episode: ${r.error} away from it`);
      sub.text(`${(r.error / norm(exact)).toFixed(1)} times the gradient's own length, `
        + `at cosine ${r.cosine}`);
    },
    2: () => {
      const r = row(16, 'plain');
      draw(sampled(16, 'plain', 777), 'var(--primary)');
      title.text(`sixteen episodes: ${r.error} away`);
      sub.text(`cosine ${r.cosine}, still further from the answer than the answer is from zero`);
    },
    3: () => {
      const r = row(16, 'baseline');
      draw(sampled(16, 'baseline', 777), 'var(--anchor)');
      title.text(`sixteen episodes, minus what each state is worth: ${r.error} away`);
      sub.text(`cosine ${r.cosine}, from subtracting a number that cannot bias anything`);
    },
  };
  handlers[0]();
  const ctrl = scrolly('#grad-scrolly .step', handlers);
  return { ...ctrl, exact };
}
