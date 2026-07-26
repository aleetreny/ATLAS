/* Pool-adjacent-violators, run by hand on 300 real calibration points.
 *
 * The dots are a deliberately crude hand-made score (elevation plus distance
 * to water, negated and rescaled: real models order these cells too well to
 * leave the algorithm any work) against what actually happened, in a balanced
 * draw so the merging is visible instead of hugging zero. Each press of
 * "merge once" performs literally one step of the algorithm: find the first
 * adjacent pair of blocks where the left average exceeds the right, and pool
 * them. When nothing violates, the staircase is the isotonic fit, and it is
 * checked against scikit-learn's answer on the same points at load time.
 */
import { drawGrid, drawAxes, axisLabels, makeChart } from '../../assets/js/chart.js';
import { pavBlocks, pavStep, pavFit } from './calkit.js';

function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function initPavWidget(data) {
  const { scores, y: yv, sk_fit } = data.pav;

  const { g, w, h } = makeChart('#pav-chart', {
    width: 560,
    height: 460,
    margin: { top: 30, right: 24, bottom: 54, left: 58 },
  });

  const x = d3.scaleLinear().domain([0, 1]).range([0, w]);
  const y = d3.scaleLinear().domain([-0.06, 1.06]).range([h, 0]);
  drawGrid(g, x, y, w, h, { yTicks: 3 });
  drawAxes(g, x, y, w, h, { yTicks: 3 });
  axisLabels(g, w, h, { x: 'the crude low-and-wet score', y: 'outcome, and the fit' });

  /* The 300 outcomes, jittered a little so coincident dots stay countable. */
  const jit = rng(7);
  g.append('g').selectAll('circle').data(scores).join('circle')
    .attr('cx', (s) => x(s))
    .attr('cy', (s, i) => y(yv[i] + (jit() - 0.5) * 0.07))
    .attr('r', 3).attr('fill', (s, i) => (yv[i] ? 'var(--primary)' : 'var(--stone)'))
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 0.5).attr('opacity', 0.7);

  const refG = g.append('g');
  const stairG = g.append('g');
  const caption = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -12).attr('text-anchor', 'middle').style('font-size', '0.62rem');

  const stats = document.querySelector('#pav-stats');
  let blocks = pavBlocks(scores, yv);
  const startCount = blocks.length;

  function stair(bs) {
    /* Horizontal segment per block, risers dotted. */
    const segs = bs.map((b) => ({ b }));
    stairG.selectAll('line.tread').data(segs).join('line').attr('class', 'tread')
      .attr('x1', (d) => x(d.b.x0) - 2).attr('x2', (d) => x(d.b.x1) + 2)
      .attr('y1', (d) => y(d.b.v)).attr('y2', (d) => y(d.b.v))
      .attr('stroke', 'var(--cosmos)').attr('stroke-width', 3).attr('stroke-linecap', 'round');
  }

  function violations(bs) {
    let n = 0;
    for (let i = 0; i + 1 < bs.length; i++) if (bs[i].v > bs[i + 1].v + 1e-12) n++;
    return n;
  }

  function render(msg) {
    stair(blocks);
    const viol = violations(blocks);
    const done = viol === 0;
    if (done) {
      refG.selectAll('circle').data(scores).join('circle')
        .attr('cx', (s) => x(s)).attr('cy', (s, i) => y(sk_fit[i]))
        .attr('r', 1.8).attr('fill', 'var(--squidink)').attr('opacity', 0.55);
    } else {
      refG.selectAll('*').remove();
    }
    caption.text(`${blocks.length} blocks, ${viol} violation${viol === 1 ? '' : 's'} left`);
    stats.innerHTML =
      `<div class="slider-label" style="line-height:1.5">${msg}</div>` +
      (done
        ? `<div class="slider-label" style="margin-top:.5rem;line-height:1.5">Monotone: this is the isotonic ` +
          `fit. The small dark dots are scikit-learn's answer on the same 300 points, and they sit exactly on ` +
          `the staircase. No formula anywhere: the shape came entirely from the data and the single rule ` +
          `"never decrease".</div>`
        : '');
  }

  document.querySelector('#pav-step').addEventListener('click', () => {
    const merged = pavStep(blocks);
    render(merged
      ? 'One merge: the first adjacent pair where the left block averaged higher than the right became one block, at their pooled mean.'
      : 'Nothing violates: the sequence is already monotone.');
  });
  document.querySelector('#pav-run').addEventListener('click', () => {
    while (pavStep(blocks)) { /* to the end */ }
    render(`Ran to completion: ${startCount} blocks pooled down to ${blocks.length}.`);
  });
  document.querySelector('#pav-reset').addEventListener('click', () => {
    blocks = pavBlocks(scores, yv);
    render(`Back to the start: every distinct score is its own block (${blocks.length} of them), most of them sitting at 0 or 1.`);
  });

  render(`Every distinct score starts as its own block: ${blocks.length} blocks, and every place a 0-block sits to the right of a 1-block is a violation of "higher score, higher rate".`);

  /* Runtime check: the page's own PAV must reproduce scikit-learn. */
  const mine = pavFit(scores, yv).fit;
  const worst = d3.max(mine.map((v, i) => Math.abs(v - sk_fit[i])));
  if (worst > 1e-4) console.warn(`browser PAV disagrees with sklearn isotonic by up to ${worst}`);
}
