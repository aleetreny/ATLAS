/* The whole pipeline, run here, with the step budget as the dial.
 *
 * The reason this widget can exist at all is the article's point. The pixel
 * article could not ship its digit model: a million and a half kilobytes of
 * float16 is not something to put in a web page, so it shipped pictures. This
 * one ships the model, because the score network is eight numbers wide and the
 * decoder is small, and the whole pipeline is under a hundred kilobytes.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';
import { decodeWeights } from '../../assets/js/halfweights.js';
import { makeCanvas, paintGray, canvasLabel } from '../../assets/js/imagery.js';

export function initLiveWidget(data, kit) {
  const M = data.meta;
  const side = M.side;
  const dim = M.pixels;
  const BUDGETS = [2, 5, 10, 25, 50, 100];
  const slider = document.querySelector('#live-slider');
  const valueTag = document.querySelector('#live-value');
  const caption = document.querySelector('#live-caption');
  const readout = document.querySelector('#live-readout');
  const st = { i: 3 };

  slider.min = 0;
  slider.max = BUDGETS.length - 1;
  slider.value = st.i;

  const N_SHOW = 24;
  const COLS = 8;
  const ROWS = Math.ceil(N_SHOW / COLS);
  const ZOOM = 7;
  const cv = makeCanvas(document.querySelector('#live-canvas'),
    COLS * side * ZOOM, ROWS * side * ZOOM);
  const chart = makeChart('#live-chart', {
    width: 400, height: 330, margin: { top: 50, right: 26, bottom: 56, left: 78 },
  });
  const { g, w, h } = chart;
  const layer = g.append('g');
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -32).attr('text-anchor', 'middle')
    .style('font-size', '12px');

  const real = decodeWeights(data.strip.real_b64, data.strip.n * dim);
  const cache = new Map();
  function draws(k) {
    if (!cache.has(k)) cache.set(k, kit.sample(N_SHOW, k, 4242));
    return cache.get(k);
  }

  function render() {
    st.i = +slider.value;
    const k = BUDGETS[st.i];
    valueTag.textContent = `${k}`;
    const { pictures, latents } = draws(k);

    const big = new Float64Array(COLS * side * ROWS * side).fill(1);
    pictures.forEach((pic, index) => {
      const r = Math.floor(index / COLS);
      const c = index % COLS;
      for (let a = 0; a < side; a++) {
        for (let b = 0; b < side; b++) {
          big[(r * side + a) * (COLS * side) + c * side + b] = 1 - pic[a * side + b];
        }
      }
    });
    paintGray(cv.ctx, big, COLS * side, ROWS * side, { zoom: ZOOM });
    canvasLabel(cv.ctx, `${k} step${k === 1 ? '' : 's'}`, 6, 14, { align: 'start', size: 11 });

    /* how far the latents have travelled: the spread of what came out, against
       the standard normal the chain started from */
    let s2 = 0;
    latents.forEach((z) => { for (let d = 0; d < kit.K; d++) s2 += z[d] * z[d]; });
    const spread = Math.sqrt(s2 / (latents.length * kit.K));
    const ink = pictures.reduce((a, p) => a + p.reduce((x, v) => x + v, 0) / dim, 0)
      / pictures.length;

    layer.selectAll('*').remove();
    const rows = data.ladder;
    const x = d3.scaleLog().domain([Math.min(...rows.map((r) => r.params)) * 0.7,
      Math.max(...rows.map((r) => r.params)) * 1.4]).range([0, w]);
    const vals = rows.map((r) => Math.abs(r.mmd2)).concat([data.floor.mmd2_max,
      Math.abs(data.ceiling.find((r) => r.k === M.k_show).mmd2)]);
    const y = d3.scaleLog().domain([Math.min(...vals) * 0.5, Math.max(...vals) * 2]).range([h, 0]);
    const decades = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
      .filter((v) => v >= y.domain()[0] && v <= y.domain()[1]);
    drawGrid(g, x, y, w, h, { xTicks: 3, yValues: decades });
    drawAxes(g, x, y, w, h, { xTicks: 3, yValues: decades, yFmt: d3.format('.0e'), xFmt: d3.format('.2s') });
    axisLabels(g, w, h, { x: 'weights', y: 'distance from real' });

    const ceil = Math.abs(data.ceiling.find((r) => r.k === M.k_show).mmd2);
    layer.append('line').attr('x1', 0).attr('x2', w).attr('y1', y(ceil)).attr('y2', y(ceil))
      .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.6).attr('stroke-dasharray', '6 4');
    layer.append('text').attr('class', 'chart-note')
      .attr('x', 2).attr('y', y(ceil) - 7).style('font-size', '11.5px')
      .attr('fill', 'var(--squidink)').text("this decoder's ceiling");
    layer.append('path')
      .attr('d', d3.line().x((r) => x(r.params)).y((r) => y(Math.abs(r.mmd2)))(rows))
      .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2.2);
    layer.selectAll('circle').data(rows).join('circle')
      .attr('cx', (r) => x(r.params)).attr('cy', (r) => y(Math.abs(r.mmd2)))
      .attr('r', (r) => (r.diffusion_params === data.net_diffusion_params ? 6 : 4))
      .attr('fill', 'var(--primary)');
    wrapLabel(title, "the ladder this widget's model sits on, and what caps it", w - 8);

    caption.textContent = `${N_SHOW} pictures made here: ${k} step${k === 1 ? '' : 's'} in a `
      + `${M.k_show} number space, then one pass through the decoder`;

    const row = data.ladder[0];
    readout.innerHTML = `
      <table class="gen-table">
        <tr>
          <th>steps</th><th>ink here</th><th>real digits</th>
          <th>latent spread</th><th>this model scored</th>
        </tr>
        <tr>
          <td><span class="value">${k}</span></td>
          <td><span class="value">${ink.toFixed(4)}</span></td>
          <td>${M.real_ink}</td>
          <td>${spread.toFixed(3)}</td>
          <td>${row.mmd2}</td>
        </tr>
      </table>
      <div class="gen-note">
        Everything above is computed in this page from
        <span class="value">${data.net.count.toLocaleString('en-US')}</span> float16 numbers: the
        score network, the decoder, and the encoder the section above uses. The pixel article
        could not do this. Its best model was
        ${data.pixel_ladder[data.pixel_ladder.length - 1].kb_half} kB and it had to ship
        photographs of its output instead, which is the same fact as the chart beside this table,
        seen from the reader's side.
        The latent spread is a sanity check rather than a result: the chain starts from a standard
        normal, whose points sit at distance one on average, and it should not end far from that.
      </div>`;
  }

  slider.addEventListener('input', render);
  render();
  return { render };
}
