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
  /* Opens at the model's own distribution, which is temperature one, and not at
     whichever temperature the judge scored highest: on this page the judge's
     ranking is confounded by ink, so opening at "the best" would be presenting
     an answer the measurement does not support. */
  slider.value = Math.max(0, temps.indexOf(1.0));

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

    /* The chart is the judge's score against the INK, not against the
       temperature, because that is the picture that answers the question the
       page actually has to answer: does this judge rank the samples, or does it
       rank how dark they are? Six points on a straight line, and the real digits
       far off past the end of it, is the whole argument in one panel. Drawn
       against temperature it looked like a result. */
    const rows = data.temperature.rows;
    const T = data.temperature;
    const xs = rows.map((r) => r.ink).concat([T.real_ink]);
    const ys = rows.map((r) => r.confidence).concat([data.meta.judge_confidence]);
    const xpad = (Math.max(...xs) - Math.min(...xs)) * 0.12;
    const ypad = (Math.max(...ys) - Math.min(...ys)) * 0.14;
    const x = d3.scaleLinear()
      .domain([Math.min(...xs) - xpad, Math.max(...xs) + xpad]).range([0, w]);
    const y = d3.scaleLinear()
      .domain([Math.min(...ys) - ypad, Math.max(...ys) + ypad]).range([h, 0]);
    drawGrid(g, x, y, w, h, { xTicks: 4, yTicks: 5 });
    drawAxes(g, x, y, w, h, { xTicks: 4, yTicks: 5 });
    /* a rotated axis label is as long as the panel is TALL, and this panel is
       214 pixels of drawing area against a label that wanted 322: the long
       version was cut off at the top. Who the judge is, is said in the prose. */
    axisLabels(g, w, h, { x: 'ink in the samples', y: "the judge's confidence" });

    layer.selectAll('*').remove();
    layer.append('path')
      .attr('d', d3.line().x((r) => x(r.ink)).y((r) => y(r.confidence))(rows))
      .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2);
    layer.selectAll('circle.t').data(rows).join('circle').attr('class', 't')
      .attr('cx', (r) => x(r.ink)).attr('cy', (r) => y(r.confidence))
      .attr('r', (r) => (r.t === t ? 6.5 : 3.4)).attr('fill', 'var(--primary)');
    /* only the two ends are labelled: six labels on a short line collide */
    [rows[0], rows[rows.length - 1]].forEach((r, i) => {
      layer.append('text').attr('class', 'chart-note')
        .attr('x', x(r.ink) + (i === 0 ? 8 : -2)).attr('y', y(r.confidence) + (i === 0 ? 14 : -10))
        .attr('text-anchor', i === 0 ? 'start' : 'end')
        .style('font-size', '11.5px').attr('fill', 'var(--primary)').text(`t = ${r.t}`);
    });
    layer.append('circle')
      .attr('cx', x(T.real_ink)).attr('cy', y(data.meta.judge_confidence)).attr('r', 6)
      .attr('fill', 'none').attr('stroke', 'var(--anchor)').attr('stroke-width', 2.2);
    layer.append('text').attr('class', 'chart-note')
      .attr('x', x(T.real_ink) - 4).attr('y', y(data.meta.judge_confidence) + 16)
      .attr('text-anchor', 'end')
      .style('font-size', '11.5px').attr('fill', 'var(--anchor)').text('real held out digits');
    wrapLabel(title, T.confounded
      ? `the judge's score tracks the ink at ${T.ink_corr}, so it is not ranking the temperatures`
      : `the judge's score does not simply track the ink (${T.ink_corr})`, w - 8);

    caption.textContent = `${COLS * ROWS} pictures, made in this page by pushing a gaussian `
      + `scaled by ${t} backwards through the flow`;

    const one = data.temperature.rows.find((r) => r.t === 1.0);
    readout.innerHTML = `
      <table class="gen-table">
        <tr>
          <th>temp</th><th>judge</th><th>ink</th>
          <th>class spread</th><th>outside</th><th>round trip</th>
        </tr>
        <tr>
          <td><span class="value">${t}</span></td>
          <td><span class="value">${row.confidence}</span></td>
          <td><span class="value">${row.ink}</span></td>
          <td>${row.class_entropy} of ${data.temperature.uniform_entropy}</td>
          <td>${(row.outside * 100).toFixed(1)}%</td>
          <td><span class="value">${trip.toExponential(1)}</span></td>
        </tr>
      </table>
      <div class="gen-note">
        The last column is the thing no other model in this module can do. The page took a held
        out digit, ran it forwards through every layer to a latent, ran it backwards again, and
        the worst pixel came back ${trip.toExponential(1)} from where it started. That is not a
        reconstruction and it is not a bound: it is the same arithmetic run twice.
        ${data.temperature.confounded
    ? `The first two columns have to be read together, and that is the point of this table. The
       judge is ${row.confidence} confident here${t === 1.0 ? ''
    : `, against ${one.confidence} at temperature one`}, and ${data.meta.judge_confidence} on
       real digits. But across the six temperatures its confidence correlates
       ${data.temperature.ink_corr} with the ink beside it, and all six are fainter than a real
       digit (${data.temperature.real_ink} there, so even the darkest sample is
       ${data.temperature.ink_ratio} of it). So the judge is not telling you which temperature
       makes better digits, it is telling you which one made more ink. The energy article
       measured the same failure on the same judge, which is why it is re-tested here rather
       than assumed away.`
    : `The judge is ${row.confidence} confident here${t === 1.0 ? ''
      : `, against ${one.confidence} at temperature one`}, and ${data.meta.judge_confidence} on
       real digits, and its ranking survives the ink check (correlation
       ${data.temperature.ink_corr}), so it can be read as a ranking.`}
        Either way the class spread is the column that catches a collapse: ${row.class_entropy}
        nats where all ten classes equally often would be
        ${data.temperature.uniform_entropy}, with the most popular taking
        ${(row.top_class_share * 100).toFixed(1)}%. A model producing one convincing digit over
        and over would score well on confidence and badly there.
      </div>`;
  }

  slider.addEventListener('input', render);
  render();
  return { render };
}
