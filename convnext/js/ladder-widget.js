/* The modernisation ladder: eleven rungs, one change each, judged against a
 * noise floor that was measured rather than assumed.
 *
 * The chart everyone draws for an ablation table is a bar per row and a story
 * about which bars are tall. This one draws the noise band too, because most
 * of these rungs do not clear it, and a ladder where seven of eleven steps are
 * indistinguishable from a different random seed is the finding, not a
 * disappointing version of one.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

const COLOUR = {
  'real gain': 'var(--primary)',
  'real loss': '#b91c1c',
  'within noise': '#8a94a6',
};

export function initLadderWidget(data) {
  const rungs = data.ladder.rungs;
  const W = 720;
  const H = 430;
  const margin = { top: 34, right: 26, bottom: 66, left: 74 };
  const { g, w, h } = makeChart('#ladder-chart', { width: W, height: H, margin });

  let mode = 'cumulative';
  let picked = rungs.length - 1;

  const bars = g.append('g');
  const axes = g.append('g');
  const grid = g.append('g');
  const marks = g.append('g');

  const x = d3.scaleBand().domain(rungs.map((r) => r.n)).range([0, w]).padding(0.22);

  function draw() {
    const step = mode === 'step';
    const vals = rungs.map((r) => (step
      ? (r.against_previous ? r.against_previous.delta : 0)
      : r.test_acc_mean));
    const lo = Math.min(...vals);
    const hi = Math.max(...vals);
    const pad = Math.max((hi - lo) * 0.18, 0.004);
    const y = step
      ? d3.scaleLinear().domain([Math.min(lo - pad, -pad), Math.max(hi + pad, pad)]).range([h, 0])
      : d3.scaleLinear().domain([lo - pad, hi + pad]).range([h, 0]);

    grid.selectAll('*').remove();
    axes.selectAll('*').remove();
    bars.selectAll('*').remove();
    marks.selectAll('*').remove();
    drawGrid(grid, x, y, w, h, { yTicks: 5 });
    drawAxes(axes, x, y, w, h, { yTicks: 5, yFmt: step ? d3.format('+.1%') : d3.format('.1%') });
    axisLabels(g, w, h, {
      x: 'rung, each one change on top of the last (click a bar)',
      y: step ? 'change from the rung below' : 'test accuracy',
    });

    /* The band every step has to clear to be called anything at all. */
    if (step) {
      const thr = rungs.find((r) => r.against_previous)?.against_previous.threshold_used || 0;
      bars.append('rect')
        .attr('x', 0).attr('width', w)
        .attr('y', y(thr)).attr('height', Math.max(1, y(-thr) - y(thr)))
        .attr('fill', '#8a94a6').attr('opacity', 0.14);
      bars.append('text')
        .attr('x', w - 4).attr('y', y(thr) - 5)
        .attr('text-anchor', 'end')
        .attr('class', 'annotation')
        .style('font-size', '11px').style('fill', '#5b6572')
        .text('anything inside this band is a different random seed');
      bars.append('line')
        .attr('x1', 0).attr('x2', w).attr('y1', y(0)).attr('y2', y(0))
        .attr('stroke', '#232f3e').attr('stroke-width', 1);
    }

    bars.selectAll('.bar')
      .data(rungs)
      .join('rect')
      .attr('class', 'bar')
      .attr('x', (r) => x(r.n))
      .attr('width', x.bandwidth())
      .attr('y', (r, i) => (step ? Math.min(y(vals[i]), y(0)) : y(vals[i])))
      .attr('height', (r, i) => (step ? Math.abs(y(vals[i]) - y(0)) : h - y(vals[i])))
      .attr('fill', (r, i) => (step
        ? (r.against_previous ? COLOUR[r.against_previous.verdict] : '#8a94a6')
        : 'var(--primary)'))
      .attr('opacity', (r, i) => (i === picked ? 1 : 0.62))
      .style('cursor', 'pointer')
      .on('click', (ev, r) => { picked = rungs.indexOf(r); draw(); });

    /* Rung numbers under the axis, and the change each one made, rotated so
       eleven of them fit without colliding. */
    /* Only the rung number goes under the axis. Eleven change descriptions
       rotated under eleven bars either collide with each other or run off
       whichever edge they are anchored towards, and the name of the selected
       change is the first thing the readout says anyway. */
    marks.selectAll('.tick-dot')
      .data(rungs)
      .join('circle')
      .attr('class', 'tick-dot')
      .attr('cx', (r) => x(r.n) + x.bandwidth() / 2)
      .attr('cy', h + 30)
      .attr('r', (r, i) => (i === picked ? 4 : 0))
      .attr('fill', 'var(--primary)');

    const r = rungs[picked];
    const vs0 = r.against_rung0;
    const prev = r.against_previous;
    const cleared = rungs.filter((q) => q.against_previous?.verdict === 'real gain').length;
    const lostN = rungs.filter((q) => q.against_previous?.verdict === 'real loss').length;
    const noiseN = rungs.length - 1 - cleared - lostN;

    document.querySelector('#ladder-readout').innerHTML =
      `<span class="bold">Rung ${r.n}: ${r.change}.</span> `
      + `${(r.test_acc_mean * 100).toFixed(2)}% on the test set across ${r.seeds.length} seed`
      + `${r.seeds.length === 1 ? '' : 's'}, ${(r.macs / 1e6).toFixed(2)} MMAC, `
      + `${r.params.toLocaleString('en-US')} parameters, trained with ${r.recipe} at ${r.lr}. `
      + (prev
        ? `Against the rung below it: ${(prev.delta * 100).toFixed(2)} points, <span class="bold">${prev.verdict}</span> `
          + `(it needed ${(prev.threshold_used * 100).toFixed(2)} points to count). `
        : 'This is the starting point, a plain residual block trained the way the previous article trained one. ')
      + `Against rung 0: ${(vs0.delta * 100).toFixed(2)} points, ${vs0.verdict}. `
      + `<br /><br />Across the whole ladder, ${cleared} step${cleared === 1 ? '' : 's'} cleared the threshold, `
      + `${lostN} lost ground, and <span class="bold">${noiseN} could not be told apart from re-rolling the seed</span>. `
      + `That is what an ablation table looks like when the noise floor is drawn on it.`;
  }

  const row = document.querySelector('#ladder-buttons');
  [['cumulative', 'where each rung stands'], ['step', 'what each change bought']]
    .forEach(([key, label], i) => {
      const b = document.createElement('button');
      b.className = `atlas-btn${i === 0 ? '' : ' ghost'}`;
      b.textContent = label;
      b.addEventListener('click', () => {
        mode = key;
        row.querySelectorAll('.atlas-btn').forEach((n) => n.classList.add('ghost'));
        b.classList.remove('ghost');
        draw();
      });
      row.appendChild(b);
    });

  draw();
}
