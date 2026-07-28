/* Five steps from a projection to a pair of learned functions.
 *
 * The last two panels run the shipped network, so what the reader sees is the
 * same arithmetic the widget below runs, on the same quantised weights.
 */
import { scrolly } from '../../assets/js/scrolly.js';
import { makeChart, wrapLabel, equalScales } from '../../assets/js/chart.js';
import { makeCanvas, paintGray } from '../../assets/js/imagery.js';
import { toRows } from '../../assets/js/embed.js';

const DIGIT = d3.scaleOrdinal(d3.schemeCategory10).domain(d3.range(10));

export function initScrollyAe(data, net) {
  const side = data.meta.side;
  const chart = makeChart('#ae-chart', {
    width: 640, height: 430, margin: { top: 58, right: 26, bottom: 66, left: 40 },
  });
  const { g, w, h } = chart;
  const body = g.append('g');
  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -38).attr('text-anchor', 'middle')
    .style('font-size', '12px').style('text-transform', 'none');
  const caption = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', h + 46).attr('text-anchor', 'middle')
    .style('font-size', '11.5px').style('text-transform', 'none');

  const Z = toRows(data.net.latent);
  const show = data.net.showcase;

  /* A strip of digits: original above, rebuilt below, drawn into one canvas so
     the comparison is a glance and not a scroll. */
  function strip(label) {
    body.selectAll('*').remove();
    const holder = body.append('foreignObject').attr('x', 0).attr('y', 10)
      .attr('width', w).attr('height', h - 10);
    const div = holder.append('xhtml:div')
      .style('display', 'flex').style('justify-content', 'center')
      .style('align-items', 'center').style('height', '100%');
    const cv = makeCanvas(div.node(), side * show.length * 5, side * 2 * 5);
    const big = new Float64Array(side * 2 * side * show.length);
    show.forEach((s, c) => {
      const px = s.pixels;
      const rec = net.decode(net.encode(px));
      for (let a = 0; a < side; a++) {
        for (let b = 0; b < side; b++) {
          big[a * (side * show.length) + c * side + b] = 1 - px[a * side + b];
          big[(side + a) * (side * show.length) + c * side + b] = 1 - rec[a * side + b];
        }
      }
    });
    paintGray(cv.ctx, big, side * show.length, side * 2, { zoom: 5 });
    return label;
  }

  function plane(colourByLabel) {
    body.selectAll('*').remove();
    const { x, y } = equalScales(Z, w, h, { pad: 1.08 });
    body.selectAll('circle').data(Z).join('circle')
      .attr('cx', (d) => x(d[0])).attr('cy', (d) => y(d[1])).attr('r', 2.6)
      .attr('fill', (d, i) => (colourByLabel ? DIGIT(data.net.labels[i]) : 'var(--squidink)'))
      .attr('opacity', 0.78);
  }

  const L = data.linear;
  const steps = [
    () => {
      strip();
      wrapLabel(title, 'ten digits, and what comes back after being squeezed through two numbers',
        w - 8);
      wrapLabel(caption, 'top row: the input. bottom row: the decoder\'s answer, from two numbers',
        w - 8);
    },
    () => {
      strip();
      wrapLabel(title, `with every activation function removed, the reconstruction error is `
        + `${L.ae_mse} against PCA's ${L.pca_mse}`, w - 8);
      wrapLabel(caption, `a ratio of ${L.mse_ratio}: the same answer, found by gradient descent `
        + 'instead of by an eigendecomposition', w - 8);
    },
    () => {
      body.selectAll('*').remove();
      const A = L.alignment;
      const cell = Math.min(96, (h - 60) / 2);
      const x0 = w / 2 - cell;
      const y0 = 24;
      for (let i = 0; i < 2; i++) {
        for (let j = 0; j < 2; j++) {
          body.append('rect').attr('x', x0 + j * cell).attr('y', y0 + i * cell)
            .attr('width', cell - 2).attr('height', cell - 2)
            .attr('fill', d3.interpolateRgb('#ffffff', '#be123c')(A[i][j]))
            .attr('stroke', 'var(--stone)');
          body.append('text').attr('x', x0 + j * cell + cell / 2 - 1)
            .attr('y', y0 + i * cell + cell / 2 + 4).attr('text-anchor', 'middle')
            .style('font-size', '14px').style('font-family', 'var(--font-mono)')
            .attr('fill', A[i][j] > 0.55 ? 'white' : 'var(--squidink)')
            .text(A[i][j].toFixed(3));
        }
      }
      ['direction 1', 'direction 2'].forEach((t, i) => {
        body.append('text').attr('x', x0 - 8).attr('y', y0 + i * cell + cell / 2 + 4)
          .attr('text-anchor', 'end').style('font-size', '11.5px')
          .style('font-family', 'var(--font-mono)').attr('fill', 'var(--squidink)').text(t);
        body.append('text').attr('x', x0 + i * cell + cell / 2).attr('y', y0 - 8)
          .attr('text-anchor', 'middle').style('font-size', '11.5px')
          .style('font-family', 'var(--font-mono)').attr('fill', 'var(--squidink)')
          .text(`PC${i + 1}`);
      });
      wrapLabel(title, 'how much each of the network\'s two directions is each principal component',
        w - 8);
      wrapLabel(caption, `1 would mean the same direction; the two subspaces differ by `
        + `${L.subspace_gap.toExponential(0)} and none of these four numbers is 1`, w - 8);
    },
    () => {
      plane(true);
      const two = data.curve.rows.find((r) => r.k === data.meta.k_show);
      wrapLabel(title, `activations back on: the same two numbers, now ${two.gain} times better `
        + 'at rebuilding a digit', w - 8);
      wrapLabel(caption, 'colour is the true digit, which the network was never shown', w - 8);
    },
    () => {
      plane(true);
      const { x, y } = equalScales(Z, w, h, { pad: 1.08 });
      const test = Z.map((p, i) => [p, i]).filter(([, i]) => data.net.is_test[i]);
      body.selectAll('circle.te').data(test).join('circle').attr('class', 'te')
        .attr('cx', (d) => x(d[0][0])).attr('cy', (d) => y(d[0][1])).attr('r', 5)
        .attr('fill', 'none').attr('stroke', 'var(--squidink)').attr('stroke-width', 1.2);
      const G = data.generalisation;
      wrapLabel(title, `${data.meta.test} of these were never in the fit, and the encoder placed `
        + 'them by running a function', w - 8);
      wrapLabel(caption, `error ${G.train_mse} on what it was fitted to, ${G.test_mse} on what it `
        + `was not, a ratio of ${G.overfit_ratio}`, w - 8);
    },
  ];

  steps[0]();
  return scrolly('#ae-scrolly .step', steps);
}
