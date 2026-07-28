/* One knob, and the three things it moves at once.
 *
 * The slider indexes the rows the generator measured, not a continuous number:
 * every position on it is a training run that happened, and the strip below is
 * eight pictures that run actually produced. The left panel is the trade the
 * objective makes (how many nats the code carries against how well a held out
 * digit comes back), the right one is whether the samples are any good, and
 * the two are not the same question, which is the entire point of the sweep.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';
import { makeCanvas, paintGray } from '../../assets/js/imagery.js';
import { unpack } from './vaekit.js';

export function initBetaWidget(data) {
  const S = data.sweep;
  const rows = S.rows;
  const side = data.meta.side;
  const px = side * side;
  const strips = unpack(S.strips_b64, S.strip_count);
  const floor = data.hole.floor;

  const holder = d3.select('#beta-chart')
    .style('display', 'flex').style('flex-wrap', 'wrap')
    .style('gap', '1.2rem').style('justify-content', 'center');
  const leftBox = holder.append('div').node();
  const rightBox = holder.append('div').node();
  const left = makeChart(leftBox, {
    width: 400, height: 330, margin: { top: 50, right: 22, bottom: 54, left: 66 },
  });
  const right = makeChart(rightBox, {
    width: 400, height: 330, margin: { top: 50, right: 22, bottom: 54, left: 80 },
  });
  left.svg.style('max-width', '410px');
  right.svg.style('max-width', '410px');

  const stripCv = makeCanvas(document.querySelector('#beta-strip'),
    side * S.strip * 3, side * 3);
  const stripCaption = d3.select('#beta-strip-caption');
  const slider = document.querySelector('#beta-slider');
  const valueTag = document.querySelector('#beta-value');
  const readout = document.querySelector('#beta-readout');
  slider.min = 0;
  slider.max = rows.length - 1;
  slider.value = rows.findIndex((r) => r.beta === 1.0);

  const lLayer = left.g.append('g');
  const rLayer = right.g.append('g');
  const lTitle = left.g.append('text').attr('class', 'chart-note')
    .attr('x', left.w / 2).attr('y', -32).attr('text-anchor', 'middle')
    .style('font-size', '12px');
  const rTitle = right.g.append('text').attr('class', 'chart-note')
    .attr('x', right.w / 2).attr('y', -32).attr('text-anchor', 'middle')
    .style('font-size', '12px');

  const rateMax = Math.max(...rows.map((r) => r.kl)) * 1.06;
  const errLo = Math.min(...rows.map((r) => r.mse));
  const errHi = Math.max(...rows.map((r) => r.mse));
  const x1 = d3.scaleLinear().domain([0, rateMax]).range([0, left.w]);
  const y1 = d3.scaleLinear()
    .domain([errLo - (errHi - errLo) * 0.16, errHi + (errHi - errLo) * 0.16])
    .range([left.h, 0]);

  const mmdVals = rows.map((r) => Math.max(r.mmd2, floor.mmd2_max));
  const x2 = d3.scalePoint().domain(rows.map((r) => String(r.beta)))
    .range([0, right.w]).padding(0.5);
  const y2 = d3.scaleLog()
    .domain([Math.min(floor.mmd2_max, Math.min(...mmdVals)) * 0.6, Math.max(...mmdVals) * 1.6])
    .range([right.h, 0]);

  function render() {
    const i = +slider.value;
    const row = rows[i];
    valueTag.textContent = `beta = ${row.beta}`;

    /* left: what the objective trades */
    drawGrid(left.g, x1, y1, left.w, left.h, { xTicks: 5, yTicks: 5 });
    drawAxes(left.g, x1, y1, left.w, left.h, { xTicks: 5, yTicks: 5, yFmt: d3.format('.3f') });
    axisLabels(left.g, left.w, left.h,
      { x: 'nats carried by the code', y: 'held out squared error' });
    lLayer.selectAll('*').remove();
    lLayer.append('path')
      .attr('d', d3.line().x((r) => x1(r.kl)).y((r) => y1(r.mse))(rows))
      .attr('fill', 'none').attr('stroke', 'var(--squidink)').attr('stroke-width', 1.4)
      .attr('opacity', 0.5);
    lLayer.selectAll('circle').data(rows).join('circle')
      .attr('cx', (r) => x1(r.kl)).attr('cy', (r) => y1(r.mse))
      .attr('r', (r, j) => (j === i ? 7 : 3.4))
      .attr('fill', (r, j) => (j === i ? 'var(--primary)' : 'var(--squidink)'))
      .attr('opacity', (r, j) => (j === i ? 1 : 0.55));
    wrapLabel(lTitle, 'more weight on the second term buys fewer nats', left.w - 8);

    /* right: whether the samples are any good */
    drawGrid(right.g, x2, y2, right.w, right.h, { xTicks: 0, yTicks: 4 });
    drawAxes(right.g, x2, y2, right.w, right.h,
      { yTicks: 4, yFmt: d3.format('.1e') });
    axisLabels(right.g, right.w, right.h,
      { x: 'weight on the second term', y: 'distance from real digits' });
    rLayer.selectAll('*').remove();
    rLayer.append('line').attr('x1', 0).attr('x2', right.w)
      .attr('y1', y2(floor.mmd2_max)).attr('y2', y2(floor.mmd2_max))
      .attr('stroke', 'var(--anchor)').attr('stroke-width', 1.4)
      .attr('stroke-dasharray', '6 4');
    rLayer.append('text').attr('class', 'chart-note')
      .attr('x', 2).attr('y', y2(floor.mmd2_max) - 7)
      .style('font-size', '11.5px')
      .attr('fill', 'var(--anchor)')
      .text('real digits against real digits');
    rLayer.append('path')
      .attr('d', d3.line().x((r) => x2(String(r.beta)))
        .y((r) => y2(Math.max(r.mmd2, floor.mmd2_max)))(rows))
      .attr('fill', 'none').attr('stroke', 'var(--squidink)').attr('stroke-width', 1.4)
      .attr('opacity', 0.5);
    rLayer.selectAll('circle').data(rows).join('circle')
      .attr('cx', (r) => x2(String(r.beta)))
      .attr('cy', (r) => y2(Math.max(r.mmd2, floor.mmd2_max)))
      .attr('r', (r, j) => (j === i ? 7 : 3.4))
      .attr('fill', (r, j) => (j === i ? 'var(--primary)' : 'var(--squidink)'))
      .attr('opacity', (r, j) => (j === i ? 1 : 0.55));
    wrapLabel(rTitle, 'and it is not the same curve', right.w - 8);

    /* the strip: eight draws from this run's own prior */
    const big = new Float64Array(side * S.strip * side);
    for (let s = 0; s < S.strip; s++) {
      const base = (i * S.strip + s) * px;
      for (let a = 0; a < side; a++) {
        for (let b = 0; b < side; b++) {
          big[a * (side * S.strip) + s * side + b] = 1 - strips[base + a * side + b];
        }
      }
    }
    paintGray(stripCv.ctx, big, side * S.strip, side, { zoom: 3 });
    stripCaption.text(`${S.strip} pictures decoded from ${S.strip} draws of the prior, `
      + `at beta = ${row.beta}`);

    const zero = rows[0];
    const best = rows.reduce((a, b) => (b.mmd2 < a.mmd2 ? b : a));
    readout.innerHTML = `
      <table class="gen-table">
        <tr>
          <th>beta</th><th>nats in the code</th><th>held out error</th>
          <th>dimensions still alive</th><th>samples against real digits</th>
          <th>a classifier separating them</th>
        </tr>
        <tr>
          <td><span class="value">${row.beta}</span></td>
          <td><span class="value">${row.kl}</span> &plusmn; ${row.kl_sd}</td>
          <td>${row.mse}</td>
          <td><span class="value">${row.active}</span> of ${data.meta.k_collapse}
              &plusmn; ${row.active_sd}</td>
          <td><span class="value">${row.mmd2}</span> &plusmn; ${row.mmd2_sd}</td>
          <td>${row.c2st} &plusmn; ${row.c2st_sd}</td>
        </tr>
      </table>
      <div class="gen-note">
        ${row.beta === 0
    ? `At zero this is not a variational autoencoder at all: the second term is gone, the
       objective is squared error through a bottleneck, and the model is the one the
       <a href="../autoencoders/">previous article</a> trained. It carries ${row.kl} nats it was
       never asked to economise, and its draws land ${(row.mmd2 / floor.mmd2_max).toFixed(0)}
       times the floor away from real digits.`
    : `${row.mmd2 <= best.mmd2
      ? `This is the closest the samples get anywhere in the sweep.`
      : `The closest samples in the sweep are at beta = ${best.beta}, at ${best.mmd2}.`}
       Against beta = 0, the rate has fallen from ${zero.kl} nats to ${row.kl} and the distance
       from real digits from ${zero.mmd2} to ${row.mmd2}, while the held out error went from
       ${zero.mse} to ${row.mse}.`}
        ${row.active < data.meta.k_collapse - 0.5
    ? ` And ${(data.meta.k_collapse - row.active).toFixed(2)} of the
        ${data.meta.k_collapse} latent dimensions now carry less than
        ${data.meta.active_threshold} nats: the model has stopped using them, which is what
        posterior collapse is when you count it instead of describing it.`
    : ` All ${data.meta.k_collapse} dimensions still carry more than
        ${data.meta.active_threshold} nats here.`}
      </div>`;
  }

  slider.addEventListener('input', render);
  render();
  return { render };
}
