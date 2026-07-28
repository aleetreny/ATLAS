/* Samples, drawn here, and the round trip that proves the model is invertible.
 *
 * Every picture in the grid is a gaussian pushed backwards through the whole
 * flow in this page. The temperature scales that gaussian before it goes in,
 * which is the one knob every flow paper turns and never explains: it does not
 * change the model at all, it changes which part of the model you are asking
 * about. The chart beside it is what the generator measured about the same
 * settings, on a pool twenty five times larger than the grid.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';
import { makeCanvas, paintGray } from '../../assets/js/imagery.js';
import { lcg } from '../../assets/js/halfweights.js';

const COLS = 6;
const ROWS = 2;

export function initTempWidget(data, glow) {
  const temps = data.temperature.rows.map((r) => r.t);
  const side = data.meta.side;
  const cv = makeCanvas(document.querySelector('#temp-canvas'), side * COLS * 8, side * ROWS * 8);
  const chart = makeChart('#temp-chart', {
    width: 400, height: 320, margin: { top: 50, right: 26, bottom: 56, left: 74 },
  });
  const { g, w, h } = chart;
  chart.svg.style('max-width', '410px');
  const slider = document.querySelector('#temp-slider');
  const valueTag = document.querySelector('#temp-value');
  const readout = document.querySelector('#temp-readout');
  const caption = document.querySelector('#temp-caption');
  slider.min = 0;
  slider.max = temps.length - 1;
  slider.value = Math.max(0, temps.indexOf(data.temperature.best));

  const layer = g.append('g');
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -32).attr('text-anchor', 'middle')
    .style('font-size', '12px');

  /* the round trip, computed once: encode a held out digit and invert it */
  const probe = data.net.probe.pixels[0];
  const back = glow.inverse(glow.forward(probe).z);
  let trip = 0;
  for (let i = 0; i < probe.length; i++) trip = Math.max(trip, Math.abs(back[i] - probe[i]));

  function paint(t) {
    const rng = lcg(101 + Math.round(t * 1000));
    const big = new Float64Array(side * COLS * side * ROWS);
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        const z = new Float64Array(glow.latentSize);
        for (let i = 0; i < z.length; i++) z[i] = rng.normal() * t;
        const px = glow.inverse(z);
        for (let a = 0; a < side; a++) {
          for (let b = 0; b < side; b++) {
            const v = Math.max(0, Math.min(1, px[a * side + b]));
            big[(r * side + a) * (side * COLS) + c * side + b] = 1 - v;
          }
        }
      }
    }
    paintGray(cv.ctx, big, side * COLS, side * ROWS, { zoom: 8 });
  }

  function render() {
    const t = temps[+slider.value];
    valueTag.textContent = `${t}`;
    paint(t);
    const row = data.temperature.rows.find((r) => r.t === t);

    const x = d3.scalePoint().domain(temps.map(String)).range([0, w]).padding(0.5);
    const conf = data.temperature.rows.map((r) => r.confidence);
    const lo = Math.min(...conf, data.meta.judge_confidence);
    const hi = Math.max(...conf, data.meta.judge_confidence);
    const y = d3.scaleLinear().domain([lo - (hi - lo) * 0.2, hi + (hi - lo) * 0.2]).range([h, 0]);
    drawGrid(g, x, y, w, h, { xValues: [], yTicks: 5 });
    drawAxes(g, x, y, w, h, { yTicks: 5 });
    axisLabels(g, w, h, { x: 'temperature', y: 'what the shared judge makes of them' });

    layer.selectAll('*').remove();
    layer.append('line').attr('x1', 0).attr('x2', w)
      .attr('y1', y(data.meta.judge_confidence)).attr('y2', y(data.meta.judge_confidence))
      .attr('stroke', 'var(--anchor)').attr('stroke-width', 1.5).attr('stroke-dasharray', '6 4');
    layer.append('text').attr('class', 'chart-note')
      .attr('x', 2).attr('y', y(data.meta.judge_confidence) - 7)
      .style('font-size', '11.5px').attr('fill', 'var(--anchor)').text('real held out digits');
    layer.append('path')
      .attr('d', d3.line().x((r) => x(String(r.t))).y((r) => y(r.confidence))(data.temperature.rows))
      .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2);
    layer.selectAll('circle').data(data.temperature.rows).join('circle')
      .attr('cx', (r) => x(String(r.t))).attr('cy', (r) => y(r.confidence))
      .attr('r', (r) => (r.t === t ? 6.5 : 3.4)).attr('fill', 'var(--primary)');
    wrapLabel(title, 'confidence is a weak measure, so read it with the next two columns', w - 8);

    caption.textContent = `${COLS * ROWS} pictures, made in this page by pushing a gaussian `
      + `scaled by ${t} backwards through the flow`;

    const one = data.temperature.rows.find((r) => r.t === 1.0);
    readout.innerHTML = `
      <table class="gen-table">
        <tr>
          <th>temperature</th><th>judge confidence</th><th>spread over the ten classes</th>
          <th>most popular class</th><th>pixels outside the range</th>
          <th>round trip, in this page</th>
        </tr>
        <tr>
          <td><span class="value">${t}</span></td>
          <td><span class="value">${row.confidence}</span>, real digits ${data.meta.judge_confidence}</td>
          <td>${row.class_entropy} nats of ${data.temperature.uniform_entropy}</td>
          <td>${(row.top_class_share * 100).toFixed(1)}%</td>
          <td>${(row.outside * 100).toFixed(1)}%</td>
          <td><span class="value">${trip.toExponential(1)}</span></td>
        </tr>
      </table>
      <div class="gen-note">
        The last column is the thing no other model in this module can do. The page took a held
        out digit, ran it forwards through every layer to a latent, ran it backwards again, and
        the worst pixel came back ${trip.toExponential(1)} from where it started. That is not a
        reconstruction and it is not a bound: it is the same arithmetic run twice.
        On the samples, the judge is ${row.confidence} confident at this temperature against
        ${one.confidence} at one and ${data.meta.judge_confidence} on real digits, but confidence
        is the weakest of the three columns, so read the class spread beside it:
        ${row.class_entropy} nats where all ten classes equally often would be
        ${data.temperature.uniform_entropy}, with the most popular taking
        ${(row.top_class_share * 100).toFixed(1)}%. A model producing one convincing digit over
        and over would score well on the first column and badly on the second.
      </div>`;
  }

  slider.addEventListener('input', render);
  render();
  return { render };
}
