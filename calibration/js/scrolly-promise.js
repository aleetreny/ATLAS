/* What a probability promises, built up on the SMOTE logistic's real test
 * claims: the exact model the previous article broke.
 *
 * Stage 0 shows the claims alone (where the model puts its probabilities);
 * then the bins, then the truth per bin, then the diagonal, then the gaps
 * become one number. Everything drawn from the real reliability bins in the
 * generator's output; nothing is a sketch.
 */
import { drawGrid, drawAxes, axisLabels, makeChart, wrapLabel } from '../../assets/js/chart.js';
import { scrolly } from '../../assets/js/scrolly.js';
import { drawDiagonal, drawReliability } from './calkit.js';

export function initScrollyPromise(data) {
  const bins = data.models.smote_logistic.raw.bins;
  const ece = data.models.smote_logistic.raw.ece;
  const nTest = data.n_test;

  const { g, w, h } = makeChart('#promise-chart', {
    width: 620,
    height: 560,
    margin: { top: 34, right: 26, bottom: 56, left: 62 },
  });

  const x = d3.scaleLinear().domain([0, 1]).range([0, w]);
  const y = d3.scaleLinear().domain([0, 1]).range([h, 0]);
  drawGrid(g, x, y, w, h);
  drawAxes(g, x, y, w, h);
  axisLabels(g, w, h, { x: 'probability the model claimed', y: 'share that really was the tree' });

  const layers = {
    bars: g.append('g'),
    diag: g.append('g'),
    gaps: g.append('g'),
    rel: g.append('g'),
  };
  const caption = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -14).attr('text-anchor', 'middle').style('font-size', '0.66rem');
  /* Two of the five captions are longer than the canvas; they wrap by measure
     and climb a lane so the second line stays inside the top margin. */
  const setCaption = (t) => {
    const lines = wrapLabel(caption, t, w - 8);
    caption.attr('y', lines > 1 ? -25 : -14);
  };

  /* The claim distribution as thin bars along the x-axis (log height,
   * because bin one holds a hundred thousand cells and bin ten holds
   * thousands). */
  const maxN = d3.max(bins, (b) => b.n);
  const barH = d3.scaleLog().domain([10, maxN]).range([6, 120]);

  function bars(show) {
    layers.bars.selectAll('rect').data(show ? bins : []).join('rect')
      .attr('x', (b) => x(b.lo) + 3)
      .attr('width', (b) => x(b.hi) - x(b.lo) - 6)
      .attr('y', (b) => h - barH(Math.max(b.n, 10)))
      .attr('height', (b) => barH(Math.max(b.n, 10)))
      .attr('fill', 'var(--primary)').attr('opacity', 0.18)
      .attr('stroke', 'var(--primary)').attr('stroke-width', 1);
    layers.bars.selectAll('text').data(show ? bins : []).join('text')
      .attr('class', 'chart-annotation')
      .attr('x', (b) => x((b.lo + b.hi) / 2))
      .attr('y', (b) => h - barH(Math.max(b.n, 10)) - 5)
      .attr('text-anchor', 'middle').style('font-size', '0.5rem').style('text-transform', 'none')
      .text((b) => b.n.toLocaleString('en-US'));
  }

  function rel(show) {
    if (!show) {
      layers.rel.selectAll('*').remove();
      return;
    }
    drawReliability(layers.rel, bins, x, y, 'var(--primary)');
  }

  function diag(show) {
    layers.diag.selectAll('*').remove();
    if (show) drawDiagonal(layers.diag, x, y, w, h);
  }

  function gaps(show) {
    layers.gaps.selectAll('line').data(show ? bins : []).join('line')
      .attr('x1', (b) => x(b.predicted)).attr('x2', (b) => x(b.predicted))
      .attr('y1', (b) => y(b.predicted)).attr('y2', (b) => y(b.observed))
      .attr('stroke', 'var(--cosmos)').attr('stroke-width', 2.4).attr('stroke-dasharray', '3 3');
  }

  const states = {
    0: () => {
      bars(true); rel(false); diag(false); gaps(false);
      setCaption(`the SMOTE model claims a probability for each of the ${nTest.toLocaleString('en-US')} test cells (bar height is log count)`);
    },
    1: () => {
      bars(true); rel(true); diag(false); gaps(false);
      setCaption('each bin: the average claim, against how often the tree was really there');
    },
    2: () => {
      bars(false); rel(true); diag(true); gaps(false);
      setCaption('an honest model would sit on the diagonal');
    },
    3: () => {
      bars(false); rel(true); diag(true); gaps(true);
      setCaption('every dashed gap is a broken promise, weighted by how many cells heard it');
    },
    4: () => {
      bars(false); rel(true); diag(true); gaps(true);
      setCaption(`expected calibration error: ${ece.toFixed(4)}, the population-weighted mean of the gaps`);
    },
  };

  scrolly('#promise-scrolly .step', states);
}
