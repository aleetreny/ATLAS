/* Six hundred walkers, running here, on the landscape the generator fitted.
 *
 * Everything in this widget is live: the energy is evaluated in the page, its
 * gradient is differentiated in the page, and the walkers move on every frame.
 * The step size indexes the sizes the generator measured, so the number in the
 * readout beside the picture is a real measurement of the same setting rather
 * than a description of it, and the three buttons are the three samplers whose
 * error the generator scored against the truth.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';
import { makeCanvas, paintGray } from '../../assets/js/imagery.js';
import { makeSampler, unpack } from './ebmkit.js';

export function initChainsWidget(data, E) {
  const G = data.model.grid;
  const A = data.model.arms.exact;
  const n = G.m;
  const energy = unpack(A.energy_b64, n * n);
  const steps = [...new Set(data.sampler.rows.map((r) => r.eps))].sort((a, b) => a - b);

  const ZOOM = Math.max(1, Math.round(340 / n));
  const cv = makeCanvas(document.querySelector('#chains-canvas'), n * ZOOM, n * ZOOM);
  const chart = makeChart('#chains-chart', {
    width: 400, height: 340, margin: { top: 50, right: 26, bottom: 56, left: 74 },
  });
  const { g, w, h } = chart;
  chart.svg.style('max-width', '410px');
  const slider = document.querySelector('#chains-slider');
  const valueTag = document.querySelector('#chains-value');
  const readout = document.querySelector('#chains-readout');
  const caption = document.querySelector('#chains-caption');
  const buttons = document.querySelector('#chains-buttons');
  slider.min = 0;
  slider.max = steps.length - 1;
  slider.value = Math.max(0, steps.indexOf(0.01));

  const st = { mode: 'ula', frame: 0 };
  const sampler = makeSampler(E, { n: 600, seed: 20250728, half: G.half });

  const btns = {};
  for (const [key, text] of [['ula', 'with noise'], ['mala', 'with the correction'],
    ['nonoise', 'without noise']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', () => {
      st.mode = key;
      sampler.reset(20250728 + steps.length * 13);
      st.frame = 0;
      draw();
    });
    buttons.appendChild(b);
    btns[key] = b;
  }
  const restart = document.createElement('button');
  restart.className = 'atlas-btn ghost';
  restart.textContent = 'start them over';
  restart.addEventListener('click', () => {
    st.seed = (st.seed || 1) + 7;
    sampler.reset(20250728 + st.seed * 977);
    st.frame = 0;
    draw();
  });
  buttons.appendChild(restart);

  /* the landscape, painted once: low energy is dark, and the ramp is on the
     square root so the ring does not vanish beside the blobs */
  const lo = A.energy_lo;
  const hi = A.energy_hi;
  const field = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const v = (energy[i * n + j] - lo) / Math.max(hi - lo, 1e-9);
      field[(n - 1 - j) * n + i] = Math.sqrt(Math.max(0, Math.min(1, v)));
    }
  }
  const ink = (t) => [255 - 160 * (1 - t), 255 - 219 * (1 - t), 255 - 237 * (1 - t)];

  function paint() {
    paintGray(cv.ctx, field, n, n, { zoom: ZOOM, colour: ink });
    const span = 2 * G.half;
    cv.ctx.save();
    cv.ctx.fillStyle = 'rgba(35,47,62,0.85)';
    sampler.points.forEach((p) => {
      const cx = ((p[0] + G.half) / span) * n * ZOOM;
      const cy = (1 - (p[1] + G.half) / span) * n * ZOOM;
      if (cx < 0 || cy < 0 || cx > n * ZOOM || cy > n * ZOOM) return;
      cv.ctx.beginPath();
      cv.ctx.arc(cx, cy, 1.6, 0, 2 * Math.PI);
      cv.ctx.fill();
    });
    cv.ctx.restore();
  }

  const layer = g.append('g');
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -32).attr('text-anchor', 'middle')
    .style('font-size', '12px');

  function chartDraw(eps) {
    const rows = data.sampler.rows;
    const x = d3.scaleLog().domain([steps[0] * 0.7, steps[steps.length - 1] * 1.4]).range([0, w]);
    const allK = rows.map((r) => Math.max(r.kl_ring, 1e-6));
    const y = d3.scaleLog().domain([Math.min(...allK) * 0.5, Math.max(...allK) * 2]).range([h, 0]);
    const decades = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
      .filter((v) => v >= y.domain()[0] && v <= y.domain()[1]);
    /* six step sizes over two decades put 0.03 and 0.05 on top of each other,
       so the grid keeps all six and the axis labels every other one */
    const labelled = steps.filter((_, i) => i % 2 === 0);
    drawGrid(g, x, y, w, h, { xValues: steps, yValues: decades });
    drawAxes(g, x, y, w, h,
      { xValues: labelled, yValues: decades, xFmt: d3.format('.2g'), yFmt: d3.format('.0e') });
    axisLabels(g, w, h, { x: 'step size', y: 'error of the sampled ring, in nats' });

    layer.selectAll('*').remove();
    [['with noise', 'ula', 'var(--primary)'], ['with the correction', 'mala', 'var(--anchor)']]
      .forEach(([label, key, colour], si) => {
        const rs = rows.filter((r) => r.sampler === key);
        layer.append('path')
          .attr('d', d3.line().x((r) => x(r.eps)).y((r) => y(Math.max(r.kl_ring, 1e-6)))(rs))
          .attr('fill', 'none').attr('stroke', colour).attr('stroke-width', 2);
        layer.selectAll(`circle.s${si}`).data(rs).join('circle')
          .attr('class', `s${si}`)
          .attr('cx', (r) => x(r.eps)).attr('cy', (r) => y(Math.max(r.kl_ring, 1e-6)))
          .attr('r', (r) => (r.eps === eps ? 6.5 : 3.2)).attr('fill', colour);
        layer.append('text').attr('class', 'chart-note')
          .attr('x', w - 4).attr('y', 12 + si * 16).attr('text-anchor', 'end')
          .style('font-size', '11.5px').attr('fill', colour).text(label);
      });
    layer.append('line').attr('x1', 0).attr('x2', w)
      .attr('y1', y(data.sampler.floor_ring)).attr('y2', y(data.sampler.floor_ring))
      .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.3).attr('stroke-dasharray', '5 4');
    layer.append('text').attr('class', 'chart-note')
      .attr('x', 2).attr('y', y(data.sampler.floor_ring) - 6)
      .style('font-size', '11.5px').text('what a perfect sampler would still show');
    wrapLabel(title, 'the error a step size costs, measured against the truth', w - 8);
  }

  function draw() {
    const eps = steps[+slider.value];
    valueTag.textContent = `${eps}`;
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', k !== st.mode));
    paint();
    chartDraw(eps);

    const row = data.sampler.rows.find((r) => r.sampler === st.mode && r.eps === eps);
    const no = data.nonoise.rows.find((r) => r.eps === eps);
    const fit = data.sampler.fit_ring;
    caption.textContent = `${sampler.points.length} walkers on the energy this page is running, `
      + `${st.frame} steps in`;
    readout.innerHTML = `
      <table class="gen-table">
        <tr>
          <th>sampler</th><th>step</th><th>error of the ring it draws</th>
          <th>width of that ring</th><th>acceptance</th><th>walkers that ever change component</th>
        </tr>
        <tr>
          <td>${st.mode === 'nonoise' ? 'no noise at all' : st.mode === 'mala' ? 'with the Metropolis correction' : 'unadjusted'}</td>
          <td><span class="value">${eps}</span></td>
          <td><span class="value">${st.mode === 'nonoise' ? no.kl_ring : row.kl_ring}</span>,
              floor ${data.sampler.floor_ring}</td>
          <td>${st.mode === 'nonoise' ? no.ring_sd : row.ring_sd} against the true
              ${data.truth.ring_width}</td>
          <td>${st.mode === 'mala' && row.accept !== null ? (row.accept * 100).toFixed(1) + '%' : 'every step is taken'}</td>
          <td>${st.mode === 'nonoise' ? 'none, they stop' : (row.switch * 100).toFixed(2) + '%'}</td>
        </tr>
      </table>
      <div class="gen-note">
        ${st.mode === 'nonoise'
    ? `With the noise removed this is not a sampler, it is gradient descent, and the walkers
       fall to the bottom and stay there. On this landscape the bottom of the ring is a circle
       rather than a point, so they do not collapse to one place: ${no.distinct_3.toLocaleString('en-US')}
       of ${data.nonoise.chains.toLocaleString('en-US')} endpoints are still distinct at three
       decimals. They collapse in the direction that matters. The ring they draw is
       ${no.ring_sd} wide against a true ${data.truth.ring_width}, its radius is
       ${no.ring_radius} against the closed form ${data.nonoise.ring_minimum}, and the error of
       the radial distribution is ${no.kl_ring} nats against ${data.sampler.rows[0].kl_ring} for
       the same chains with the noise put back.`
    : st.mode === 'mala'
      ? `The Metropolis correction proposes the same step and then accepts it with the
         probability that makes the chain exact. At this step it accepts
         ${(row.accept * 100).toFixed(1)}% of the time and the error is ${row.kl_ring}, against
         ${data.sampler.rows.find((r) => r.sampler === 'ula' && r.eps === eps).kl_ring} for the
         unadjusted version at the same step. The cost is one extra energy evaluation per
         walker per step, and it is why the correction exists.`
      : `Every step is taken, so the chain is not sampling from this energy: it is sampling from
         a distribution near it, and the gap closes as the step shrinks. Fitted over the
         ${fit.points} step sizes above the floor, from ${fit.eps_min} to ${fit.eps_max}, the
         error falls as the step to the power <span class="value">${fit.exponent}</span>
         (fit quality ${fit.r2}), so halving the step is worth about
         ${(2 ** fit.exponent).toFixed(1)} times less error and about twice the work.`}
      </div>`;
  }

  let raf = null;
  function loop() {
    const eps = steps[+slider.value];
    for (let i = 0; i < 3; i++) sampler.step(eps, st.mode);
    st.frame += 3;
    paint();
    caption.textContent = `${sampler.points.length} walkers on the energy this page is running, `
      + `${st.frame} steps in`;
    raf = requestAnimationFrame(loop);
  }

  slider.addEventListener('input', () => { st.frame = 0; sampler.reset(20250728); draw(); });
  draw();
  /* the chains only run while the widget is on screen: an animation frame loop
     left running behind a scroll is a battery bug, not a feature */
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting && raf === null) raf = requestAnimationFrame(loop);
      else if (!e.isIntersecting && raf !== null) { cancelAnimationFrame(raf); raf = null; }
    });
  }, { threshold: 0.05 });
  io.observe(document.querySelector('#chains-widget'));
  return st;
}
