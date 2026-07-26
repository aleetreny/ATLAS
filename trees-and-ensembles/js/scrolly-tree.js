/* Growing one tree, six steps, drawn as the boxes it actually is.
 * Handlers are idempotent state declarations so scrolling back works for free.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { scrolly } from '../../assets/js/scrolly.js';
import { drawBoxes, drawPoints } from './boxkit.js';

export function initScrollyTree(data) {
  const D = data.two_d;
  const { g, w, h } = makeChart('#tree-chart', {
    width: 620,
    height: 540,
    margin: { top: 34, right: 26, bottom: 58, left: 68 },
  });

  const x = d3.scaleLinear().domain(D.bounds[0]).range([0, w]);
  const y = d3.scaleLinear().domain(D.bounds[1]).range([h, 0]);
  drawGrid(g, x, y, w, h);
  drawAxes(g, x, y, w, h);
  axisLabels(g, w, h, {
    x: D.features[0].replace(/_/g, ' '),
    y: D.features[1].replace(/_/g, ' '),
  });

  const clip = 'tree-clip';
  g.append('clipPath').attr('id', clip).append('rect').attr('x', 0).attr('y', 0)
    .attr('width', w).attr('height', h);
  const boxG = g.append('g').attr('clip-path', `url(#${clip})`);
  const dotG = g.append('g');
  drawPoints(dotG, D.train, x, y);

  const caption = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -14).attr('text-anchor', 'middle').style('font-size', '0.66rem');

  const byDepth = Object.fromEntries(D.depths.map((r) => [r.depth, r]));
  const show = (depth, label) => {
    drawBoxes(boxG, depth === null ? [] : D.boxes[String(depth)], x, y, { duration: 430 });
    caption.text(label);
  };

  scrolly('#tree-scrolly .step', {
    0: () => show(null, `${D.train.length} training biopsies, no model yet`),
    1: () => show(1, `one split: ${D.first_split.feature.replace(/_/g, ' ')} at ${D.first_split.threshold}`),
    2: () => show(2, `${D.boxes['2'].length} boxes, held out ${byDepth[2].test.toFixed(4)}`),
    3: () => show(3, `${D.boxes['3'].length} boxes, held out ${byDepth[3].test.toFixed(4)}`),
    4: () => show(5, `${D.boxes['5'].length} boxes, train ${byDepth[5].train.toFixed(4)} but held out ${byDepth[5].test.toFixed(4)}`),
    5: () => show(12, `${D.boxes['12'].length} boxes, train ${byDepth[12].train.toFixed(4)}, held out ${byDepth[12].test.toFixed(4)}`),
  });

  /* Runtime checks on the two numbers the steps assert. The article claims the
   * unpruned tree is perfect on the training set and worst on held-out data,
   * and both should be measured rather than trusted. */
  if (byDepth[12].train !== 1) {
    console.warn(`unpruned tree does not reach a perfect training score: ${byDepth[12].train}`);
  }
  const worst = D.depths.reduce((a, b) => (b.test < a.test ? b : a));
  if (worst.test !== byDepth[12].test) {
    console.warn(`the worst held-out depth is ${worst.depth}, not the unpruned tree`);
  }
}
