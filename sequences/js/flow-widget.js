/* Two measurements of memory, on one pair of axes each.
 *
 * The first view is the gradient: how much the input at a step still moves the
 * state some number of steps later, in each of the three cells, measured on
 * the trained models rather than on freshly initialised ones. Initialised
 * weights would make this a picture of the initialisation.
 *
 * The second view is the task: remember one symbol across a growing number of
 * distractors, and report where each architecture stops being able to. The two
 * are worth putting behind one button because the second is what the first is
 * supposed to predict, and whether it does is the question.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { seriesLabel, legend, logTicks } from '../../assets/js/textdraw.js';

const CELLS = [
  { key: 'rnn', label: 'plain RNN', colour: 'var(--cosmos)' },
  { key: 'gru', label: 'GRU', colour: 'var(--anchor)' },
  { key: 'lstm', label: 'LSTM', colour: 'var(--primary)' },
];
const VIEWS = [
  { key: 'flow', label: 'gradient, after training' },
  { key: 'flow0', label: 'gradient, before training' },
  { key: 'copy', label: 'how far the memory reaches' },
];

/* d3's default on a log scale hands back 0.30000000000000004. Powers of ten
 * spelled out, everything else to two significant figures. */
const tick = (v) => {
  if (v >= 1) return String(Math.round(v));
  if (v >= 0.01) return v.toFixed(2);
  return v.toExponential(0);
};

export function initFlowWidget(data) {
  let view = 'flow';
  const chart = makeChart('#flow-chart', {
    width: 640, height: 400, margin: { top: 44, right: 30, bottom: 58, left: 72 },
  });
  const { g, w, h } = chart;
  const readout = d3.select('#flow-readout');

  d3.select('#flow-views').selectAll('button').data(VIEWS).join('button')
    .attr('class', (d) => `atlas-btn${d.key === view ? '' : ' ghost'}`)
    .text((d) => d.label)
    .on('click', (event, d) => {
      view = d.key;
      d3.select('#flow-views').selectAll('button')
        .attr('class', (q) => `atlas-btn${q.key === view ? '' : ' ghost'}`);
      render();
    });

  function drawFlow(which) {
    const src = which === 'flow0' ? (data.flow_untrained || data.flow) : data.flow;
    const series = CELLS.map((c) => ({ ...c, f: src[c.key] })).filter((s) => s.f);
    const maxD = Math.max(...series.map((s) => s.f.distances.length));
    const lo = Math.max(1e-8, Math.min(...series.flatMap((s) => s.f.relative.filter((v) => v > 0))));
    const x = d3.scaleLinear().domain([1, maxD]).range([0, w]);
    const y = d3.scaleLog().domain([lo, 1.2]).range([h, 0]);
    const yv = logTicks(lo, 1.2);

    g.append('g').attr('class', 'grid')
      .call((s) => drawGrid(s, x, y, w, h, { xTicks: 6, yValues: yv }));
    const axes = g.append('g');
    drawAxes(axes, x, y, w, h, { xTicks: 6, yValues: yv, yFmt: tick });
    axisLabels(axes, w, h, {
      x: 'steps back from the end of the sentence',
      y: 'influence still carried',
    });

    /* Half of the starting influence, drawn, so "half life" is readable off
     * the picture rather than only stated in the readout. */
    g.append('line')
      .attr('x1', 0).attr('x2', w).attr('y1', y(0.5)).attr('y2', y(0.5))
      .attr('stroke', 'var(--squidink)').attr('stroke-width', 1)
      .attr('stroke-dasharray', '3 3').attr('opacity', 0.45);
    seriesLabel(g, w, y(0.5) - 6, 'half', { anchor: 'end', size: 10.5, opacity: 0.6 });

    series.forEach((s) => {
      const pts = s.f.distances.map((d, i) => ({ d, v: s.f.relative[i] }))
        .filter((p) => p.v > 0);
      g.append('path').datum(pts)
        .attr('fill', 'none').attr('stroke', s.colour).attr('stroke-width', 2.2)
        .attr('d', d3.line().x((p) => x(p.d)).y((p) => y(p.v)));
    });
    legend(g, series.map((s) => ({ label: s.label, colour: s.colour, shape: 'line' })),
      { x0: 0, y0: -22, gap: 130, size: 10.5 });

    const rows = series.map((s) => `<span class="bold">${s.label}</span> falls to half in `
      + `${s.f.half_life ? `${s.f.half_life} step${s.f.half_life === 1 ? '' : 's'}` : 'more than the sentence is long'}`
      + `${s.f.at_30 !== null && s.f.at_30 !== undefined
        ? ` and is at <span class="value">${s.f.at_30.toExponential(1)}</span> of its starting `
          + 'influence thirty steps back' : ''}`);
    readout.html(
      `<div>${rows.join('. ')}.</div>`
      + (which === 'flow0'
        ? `<div>These are the <span class="bold">untrained</span> networks, with the same initial `
          + `weights the trained ones started from. This is the regime the vanishing gradient `
          + `argument is actually about: it is a claim about what backpropagation can carry before `
          + `anything has been learned.</div>`
        : `<div>These are the <span class="bold">trained</span> networks, and the three curves `
          + `land almost on top of each other. That is not the probe failing. Trained on a task `
          + `whose answers are local, all three learned to be local, so what this view measures is `
          + `what the task asked for rather than what the cell could have done. The view before `
          + `training separates them, and the copy task separates them further.</div>`)
      + `<div>Every curve is divided by its own value one step back, so what is being compared is `
      + `the <span class="bold">shape</span> of the decay and not its scale: three models with `
      + `different starting magnitudes would otherwise be three parallel lines and the picture `
      + `would be measuring that instead. Measured on the trained models over `
      + `<span class="value">${series[0].f.sentences}</span> long sentences, taking the median at `
      + `each distance.</div>`
    );
  }

  function drawCopy() {
    const series = CELLS.map((c) => ({ ...c, rows: data.copy[c.key] })).filter((s) => s.rows);
    const ds = data.copy_distances;
    const x = d3.scalePoint().domain(ds).range([0, w]).padding(0.5);
    const y = d3.scaleLinear().domain([0, 1.02]).range([h, 0]);
    g.append('g').attr('class', 'grid')
      .call((s) => drawGrid(s, x, y, w, h, { xTicks: 0, yTicks: 5 }));
    const axes = g.append('g');
    drawAxes(axes, x, y, w, h, { yTicks: 5, yFmt: (v) => v.toFixed(1) });
    axisLabels(axes, w, h, {
      x: 'distractors between the symbol and the question',
      y: 'how often the symbol is recalled',
    });

    const chance = series[0].rows[0].chance;
    g.append('line')
      .attr('x1', 0).attr('x2', w).attr('y1', y(chance)).attr('y2', y(chance))
      .attr('stroke', 'var(--squidink)').attr('stroke-width', 1)
      .attr('stroke-dasharray', '3 3').attr('opacity', 0.5);
    seriesLabel(g, w, y(chance) - 6, 'guessing', { anchor: 'end', size: 10.5, opacity: 0.6 });

    series.forEach((s) => {
      g.append('path').datum(s.rows)
        .attr('fill', 'none').attr('stroke', s.colour).attr('stroke-width', 2.2)
        .attr('d', d3.line().x((r) => x(r.distance)).y((r) => y(r.acc)));
      s.rows.forEach((r) => {
        g.append('circle').attr('cx', x(r.distance)).attr('cy', y(r.acc)).attr('r', 3.6)
          .attr('fill', s.colour);
      });
    });
    legend(g, series.map((s) => ({ label: s.label, colour: s.colour, shape: 'line' })),
      { x0: 0, y0: -22, gap: 130, size: 10.5 });

    const breaks = series.map((s) => {
      const bad = s.rows.find((r) => r.acc < 0.6);
      return { s, at: bad ? bad.distance : null };
    });
    readout.html(
      `<div>${breaks.map((b) => `<span class="bold">${b.s.label}</span> `
        + (b.at === null
          ? `still solves it at ${ds[ds.length - 1]} distractors`
          : `drops below six in ten at ${b.at} distractors`)).join(', ')}.</div>`
      + `<div>There is nothing in this task but memory: the answer is the first symbol, `
      + `everything after it is drawn from a different alphabet and carries no information, and `
      + `the last position asks the question. A model with no memory at all scores `
      + `<span class="value">${chance.toFixed(3)}</span>. All three were given the same parameter `
      + `budget, with the hidden width adjusted per cell so that four gates and one gate do not `
      + `mean four times the model.</div>`
    );
  }

  function render() {
    g.selectAll('*').remove();
    if (view === 'copy') drawCopy();
    else drawFlow(view);
  }

  render();
  return { render };
}
