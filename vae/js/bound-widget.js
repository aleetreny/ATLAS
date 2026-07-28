/* Two dimensions, where the thing the bound bounds can actually be computed.
 *
 * Everywhere else in this module the marginal likelihood is an integral nobody
 * does, so the bound is all there is and its looseness is unknown. Here the
 * data lives in the plane, the latent lives in the plane, and the integral is
 * a sum: the generator computed it two independent ways and they agree to the
 * number in the readout. So the gap between the bound and the truth stops
 * being a worry and becomes a measurement, and the importance weighted bound
 * can be watched closing it one order of magnitude at a time.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';
import { makeCanvas, paintGray, canvasLabel } from '../../assets/js/imagery.js';
import { unpack } from './vaekit.js';

export function initBoundWidget(data) {
  const T = data.toy;
  const P = T.picture;
  const n = P.n;
  const truth = unpack(P.truth_b64, n * n);
  const model = unpack(P.model_b64, n * n);

  const ZOOM = Math.max(1, Math.round(360 / n));
  const cv = makeCanvas(document.querySelector('#bound-picture'), n * ZOOM, n * ZOOM);
  const chart = makeChart('#bound-chart', {
    width: 400, height: 340, margin: { top: 50, right: 26, bottom: 56, left: 70 },
  });
  const { g, w, h } = chart;
  chart.svg.style('max-width', '410px');
  const buttons = document.querySelector('#bound-buttons');
  const readout = document.querySelector('#bound-readout');

  const st = { view: 'truth' };
  const btns = {};
  for (const [key, text] of [['truth', 'the truth'], ['model', 'what it learned'],
    ['diff', 'where they differ']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', () => { st.view = key; render(); });
    buttons.appendChild(b);
    btns[key] = b;
  }

  const layer = g.append('g');
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -32).attr('text-anchor', 'middle')
    .style('font-size', '12px');

  /* A density is positive and very peaked, so the ramp is on its square root:
   * monotone, so darker still means more, and the ring does not vanish next to
   * the blobs. */
  const peak = Math.sqrt(P.peak);
  const ink = (t) => [255 - 131 * t, 255 - 210 * t, 255 - 237 * t];      // white to #7c2d12
  const anchor = (t) => [255 - 255 * t, 255 - 206 * t, 255 - 126 * t];   // white to #003181

  function paint() {
    const vals = new Float64Array(n * n);
    let colour;
    if (st.view === 'diff') {
      let m = 0;
      for (let i = 0; i < n * n; i++) {
        m = Math.max(m, Math.abs(Math.sqrt(model[i]) - Math.sqrt(truth[i])));
      }
      for (let i = 0; i < n * n; i++) {
        vals[i] = (Math.sqrt(model[i]) - Math.sqrt(truth[i])) / (m || 1);
      }
      colour = (v) => (v >= 0 ? ink(Math.min(1, v)) : anchor(Math.min(1, -v)));
    } else {
      const src = st.view === 'truth' ? truth : model;
      for (let i = 0; i < n * n; i++) vals[i] = Math.min(1, Math.sqrt(src[i]) / peak);
      colour = (v) => ink(v);
    }
    /* the grid is indexed by x then y, and a canvas by row then column */
    const flip = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) flip[(n - 1 - j) * n + i] = vals[i * n + j];
    }
    paintGray(cv.ctx, flip, n, n, { zoom: ZOOM, colour });
    if (st.view === 'truth') {
      cv.ctx.save();
      cv.ctx.fillStyle = 'rgba(35,47,62,0.75)';
      const span = P.hi - P.lo;
      P.sample.forEach((p) => {
        const cx = ((p[0] - P.lo) / span) * n * ZOOM;
        const cy = (1 - (p[1] - P.lo) / span) * n * ZOOM;
        cv.ctx.beginPath();
        cv.ctx.arc(cx, cy, 1.1, 0, 2 * Math.PI);
        cv.ctx.fill();
      });
      cv.ctx.restore();
    }
    canvasLabel(cv.ctx, `${P.lo}`, 4, n * ZOOM - 5, { align: 'start', size: 10 });
    canvasLabel(cv.ctx, `${P.hi}`, n * ZOOM - 4, n * ZOOM - 5, { align: 'end', size: 10 });
  }

  function render() {
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', k !== st.view));
    paint();

    const rows = T.bounds;
    const x = d3.scaleLog().domain([0.8, rows[rows.length - 1].k * 1.6]).range([0, w]);
    const lo = Math.min(...rows.map((r) => r.bound), T.exact, T.ceiling);
    const hi = Math.max(...rows.map((r) => r.bound), T.exact, T.ceiling);
    const pad = (hi - lo) * 0.12 + 1e-6;
    const y = d3.scaleLinear().domain([lo - pad, hi + pad]).range([h, 0]);
    const ticks = rows.map((r) => r.k);

    drawGrid(g, x, y, w, h, { xValues: ticks, yTicks: 5 });
    drawAxes(g, x, y, w, h, { xValues: ticks, yTicks: 5, xFmt: d3.format(',d') });
    axisLabels(g, w, h, { x: 'importance samples per row', y: 'nats per held out row' });

    layer.selectAll('*').remove();
    [['exact', T.exact, 'var(--primary)', 'the exact log density'],
      ['ceiling', T.ceiling, 'var(--anchor)', 'the truth itself']]
      .forEach(([, v, colour, label], idx) => {
        layer.append('line').attr('x1', 0).attr('x2', w).attr('y1', y(v)).attr('y2', y(v))
          .attr('stroke', colour).attr('stroke-width', 1.5).attr('stroke-dasharray', '6 4');
        layer.append('text').attr('class', 'chart-note')
          .attr('x', 2).attr('y', y(v) + (idx === 0 ? -7 : 14))
          .style('font-size', '11.5px')
          .attr('fill', colour).text(label);
      });
    layer.append('path')
      .attr('d', d3.line().x((r) => x(r.k)).y((r) => y(r.bound))(rows))
      .attr('fill', 'none').attr('stroke', 'var(--squidink)').attr('stroke-width', 1.8);
    layer.selectAll('circle').data(rows).join('circle')
      .attr('cx', (r) => x(r.k)).attr('cy', (r) => y(r.bound)).attr('r', 4)
      .attr('fill', 'var(--squidink)');
    wrapLabel(title, 'the bound, and the number it is a bound on', w - 8);

    const one = rows.find((r) => r.k === 1);
    const last = rows[rows.length - 1];
    readout.innerHTML = `
      <table class="gen-table">
        <tr>
          <th>the bound, one sample</th><th>the bound, ${last.k.toLocaleString('en-US')} samples</th>
          <th>the exact log density</th><th>the truth's own</th>
          <th>the gap the bound leaves</th><th>KL from the truth</th>
        </tr>
        <tr>
          <td>${one.bound}</td>
          <td>${last.bound}</td>
          <td><span class="value">${T.exact}</span></td>
          <td>${T.ceiling}</td>
          <td><span class="value">${T.gap}</span> nats</td>
          <td><span class="value">${T.kl_to_truth}</span> nats</td>
        </tr>
      </table>
      <div class="gen-note">
        ${st.view === 'truth'
    ? `The density this module is fitted against, drawn on the square its numbers are measured
       on, with ${P.sample.length} of the held out rows on top of it. It is written in closed
       form, so its own mean log density is exactly ${T.ceiling} nats and nothing can do better.
       The hole in the middle of the ring holds ${P.hole.truth} of the mass inside radius
       ${P.hole.radius}, and it is the reason this shape is here: it is the piece the flows
       article cannot reproduce.`
    : st.view === 'model'
      ? `What ${T.epochs} epochs of the bound produced. Its integral over this square is
         ${T.guards.model_mass}, which is the check that the number being called a density is
         one, and it puts ${P.hole.model} inside the hole against the truth's ${P.hole.truth}.`
      : `The difference between the two square roots, in this article's accent where the fit
         puts too much and in blue where it puts too little. A variational autoencoder is
         trained on a lower bound, so nothing forces it to be tight anywhere in particular,
         and the picture shows where it chose to spend.`}
        The bound at one sample is ${one.bound} against an exact ${T.exact}: a gap of
        ${T.gap} nats that no amount of looking at the loss curve would ever reveal.
        ${last.k.toLocaleString('en-US')} importance samples close it to ${last.gap}.
      </div>`;
  }

  render();
  return st;
}
