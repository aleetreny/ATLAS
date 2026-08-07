/* The reference run, pull by pull.
 *
 * Not a diagram of a bandit: the run in the file, replayed here by the same
 * code that produced it, so the accident in the third step is an accident that
 * happened rather than one drawn in to make a point.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { scrolly } from '../../assets/js/scrolly.js';
import { runBandit } from './banditkit.js';

export function initPullScrolly(data) {
  const R = data.reference;
  const arms = R.arms;
  const k = arms.length;
  const run = runBandit(arms, { seed: R.config.seed, pulls: R.config.pulls,
    eps: R.config.eps });

  const chart = makeChart('#pull-chart', {
    width: 640, height: 420, margin: { top: 54, right: 40, bottom: 96, left: 62 },
  });
  const { g, w, h, margin } = chart;
  const x = d3.scalePoint().domain(d3.range(k).map(String)).range([0, w]).padding(0.6);
  /* an estimate can be 1.00 after a single winning pull, so the axis has to
     reach past it or the label sits outside the canvas */
  const y = d3.scaleLinear().domain([0, 1.12]).range([h, 0]);
  drawGrid(g, x, y, w, h, { yTicks: 4 });
  drawAxes(g, x, y, w, h, { yTicks: 4, xValues: [] });
  axisLabels(g, w, h, { y: 'chance of a win', leftBudget: margin.left });

  /* the truth, which the player never sees, drawn as a hollow marker */
  g.selectAll('line.truth').data(arms).join('line').attr('class', 'truth')
    .attr('x1', (_, i) => x(String(i)) - 26).attr('x2', (_, i) => x(String(i)) + 26)
    .attr('y1', (d) => y(d)).attr('y2', (d) => y(d))
    .attr('stroke', '#8e9aa6').attr('stroke-width', 1.6).attr('stroke-dasharray', '4 3');
  const bars = g.append('g');
  const labels = g.append('g');
  const title = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -34)
    .style('font-size', '12px');
  const sub = g.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -14)
    .style('font-size', '11.5px');
  d3.range(k).forEach((i) => {
    g.append('text').attr('class', 'chart-note').attr('x', x(String(i)))
      .attr('y', h + 18).attr('text-anchor', 'middle').style('font-size', '10.5px')
      .text(`machine ${i + 1}`);
    g.append('text').attr('class', 'chart-note').attr('x', x(String(i)))
      .attr('y', h + 34).attr('text-anchor', 'middle').style('font-size', '10.5px')
      .text(`truth ${arms[i]}`);
  });

  function draw(upTo, note) {
    const counts = new Array(k).fill(0);
    const sums = new Array(k).fill(0);
    run.history.slice(0, upTo).forEach((p) => { counts[p.arm] += 1; sums[p.arm] += p.reward; });
    const est = counts.map((c, i) => (c ? sums[i] / c : 0));
    bars.selectAll('rect').data(est).join('rect')
      .attr('x', (_, i) => x(String(i)) - 20).attr('width', 40)
      .transition('bars').duration(500)
      .attr('y', (d) => y(d)).attr('height', (d) => h - y(d))
      .attr('fill', (_, i) => (counts[i] === Math.max(...counts) ? 'var(--primary)' : '#c9ced0'));
    labels.selectAll('text').data(est).join('text').attr('class', 'chart-note')
      .attr('x', (_, i) => x(String(i))).attr('y', (d) => y(d) - 7)
      .attr('text-anchor', 'middle').style('font-size', '10.5px')
      .text((d, i) => (counts[i] ? `${d.toFixed(2)} in ${counts[i]}` : 'never pulled'));
    const regret = upTo ? run.history[upTo - 1].regret : 0;
    title.text(`after ${upTo} pulls: regret ${regret.toFixed(2)}`);
    sub.text(note);
  }

  /* the first pull of the best machine after the greedy phase has locked on */
  const firstLock = run.history.findIndex((p, i) => i > k + 5 && p.arm !== run.history[k + 5].arm);
  const bestArm = arms.indexOf(Math.max(...arms));
  const firstBest = run.history.findIndex((p, i) => i > k && p.arm === bestArm);

  const handlers = {
    0: () => draw(k, `one pull each: ${run.history.slice(0, k).filter((p) => p.reward).length} `
      + `wins out of ${k}, which is all anybody knows`),
    1: () => draw(Math.max(k + 1, Math.min(firstBest, 40)),
      'the estimate with the luckiest start gets pulled again, and again'),
    2: () => draw(Math.min(firstBest + 40, run.history.length),
      `the best machine is first tried on purpose at pull ${firstBest + 1}`),
    3: () => draw(run.history.length,
      `${run.counts[bestArm]} of ${R.config.pulls} pulls went to the best machine`),
  };
  handlers[0]();
  const ctrl = scrolly('#pull-scrolly .step', handlers);
  return { ...ctrl, run, firstBest, firstLock };
}
