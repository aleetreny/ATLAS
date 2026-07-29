/* The reverse process, run here, with the step budget as the dial.
 *
 * The chain is the model. This widget runs it in the page from the same float16
 * weights the generator measured, so the pictures are not a recording: change
 * the budget and the arithmetic happens again. The two samplers are the same
 * trained network with one term switched off, which is the distinction the
 * article keeps making and the one most treatments blur.
 *
 * The chart beside it is the measurement, and it carries its floor: two pools
 * of REAL draws score this far apart from each other, so anything inside that
 * band is not a difference.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';
import { makeCanvas, canvasLabel } from '../../assets/js/imagery.js';

export function initReverseWidget(data, kit) {
  const R = data.sampling;
  const BUDGETS = [2, 5, 10, 25, 50, 100, 250, 1000];
  const slider = document.querySelector('#reverse-slider');
  const valueTag = document.querySelector('#reverse-value');
  const caption = document.querySelector('#reverse-caption');
  const readout = document.querySelector('#reverse-readout');
  const buttons = document.querySelector('#reverse-buttons');
  const st = { mode: 'deterministic', i: 4 };

  slider.min = 0;
  slider.max = BUDGETS.length - 1;
  slider.value = st.i;

  const btns = {};
  for (const [key, text] of [['deterministic', 'deterministic'],
    ['ancestral', 'ancestral, with the noise put back']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', () => { st.mode = key; render(); });
    buttons.appendChild(b);
    btns[key] = b;
  }

  const SIDE = 340;
  const HALF = 4.6;
  const cv = makeCanvas(document.querySelector('#reverse-canvas'), SIDE, SIDE);
  const chart = makeChart('#reverse-chart', {
    width: 400, height: 330, margin: { top: 50, right: 26, bottom: 56, left: 78 },
  });
  const { g, w, h } = chart;
  /* the width is the column's, set in the page's stylesheet: see .chart-half */
  const layer = g.append('g');
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -32).attr('text-anchor', 'middle')
    .style('font-size', '12px');

  /* the draws the page makes: enough to see the shape, few enough that a
     thousand step chain still finishes while a reader is looking at it */
  const N_DRAW = 900;
  const cache = new Map();
  function draws(k, mode) {
    const key = `${k}:${mode}`;
    if (!cache.has(key)) cache.set(key, kit.reverse(N_DRAW, k, 20250728, mode));
    return cache.get(key);
  }

  function paint(k, mode) {
    const { frames } = draws(k, mode);
    const last = frames[frames.length - 1];
    const px = (x) => ((x + HALF) / (2 * HALF)) * SIDE;
    const py = (y) => (1 - (y + HALF) / (2 * HALF)) * SIDE;
    cv.ctx.clearRect(0, 0, SIDE, SIDE);
    cv.ctx.fillStyle = '#f7f8f8';
    cv.ctx.fillRect(0, 0, SIDE, SIDE);
    /* the circle every hole number on this page is about, the same one the
       flows article drew */
    cv.ctx.save();
    cv.ctx.strokeStyle = '#8a4b12';
    cv.ctx.lineWidth = 1.4;
    cv.ctx.setLineDash([5, 4]);
    cv.ctx.beginPath();
    cv.ctx.arc(px(0), py(0), (1.2 / (2 * HALF)) * SIDE, 0, Math.PI * 2);
    cv.ctx.stroke();
    cv.ctx.restore();
    cv.ctx.fillStyle = 'rgba(103,69,153,0.72)';
    let inHole = 0;
    for (let i = 0; i < N_DRAW; i++) {
      const x = last[i * 2];
      const y = last[i * 2 + 1];
      if (x * x + y * y <= 1.44) inHole++;
      const cx = px(x);
      const cy = py(y);
      if (cx < -4 || cy < -4 || cx > SIDE + 4 || cy > SIDE + 4) continue;
      cv.ctx.beginPath();
      cv.ctx.arc(cx, cy, 1.5, 0, Math.PI * 2);
      cv.ctx.fill();
    }
    canvasLabel(cv.ctx, `${-HALF}`, 4, SIDE - 5, { align: 'start', size: 10 });
    canvasLabel(cv.ctx, `${HALF}`, SIDE - 4, SIDE - 5, { align: 'end', size: 10 });
    return inHole;
  }

  function render() {
    st.i = +slider.value;
    const k = BUDGETS[st.i];
    valueTag.textContent = `${k}`;
    Object.entries(btns).forEach(([key, b]) => b.classList.toggle('ghost', key !== st.mode));
    const inHole = paint(k, st.mode);
    layer.selectAll('*').remove();

    const rows = R.rows.filter((r) => r.mode === st.mode);
    const x = d3.scaleLog().domain([1.6, 1300]).range([0, w]);
    const vals = R.rows.map((r) => Math.abs(r.mmd2)).concat([R.floor.mmd2_max]);
    const y = d3.scaleLog().domain([Math.min(...vals) * 0.5, Math.max(...vals) * 2]).range([h, 0]);
    const decades = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
      .filter((v) => v >= y.domain()[0] && v <= y.domain()[1]);
    drawGrid(g, x, y, w, h, { xValues: BUDGETS, yValues: decades });
    drawAxes(g, x, y, w, h, { xValues: BUDGETS, yValues: decades, yFmt: d3.format('.0e') });
    axisLabels(g, w, h, { x: 'steps allowed', y: 'distance from real draws' });

    /* the floor, drawn as a band: below it there is nothing to see */
    layer.append('rect').attr('x', 0).attr('y', y(R.floor.mmd2_max))
      .attr('width', w).attr('height', Math.max(0, h - y(R.floor.mmd2_max)))
      .attr('fill', 'var(--anchor)').attr('opacity', 0.1);
    layer.append('line').attr('x1', 0).attr('x2', w)
      .attr('y1', y(R.floor.mmd2_max)).attr('y2', y(R.floor.mmd2_max))
      .attr('stroke', 'var(--anchor)').attr('stroke-width', 1.5).attr('stroke-dasharray', '6 4');
    layer.append('text').attr('class', 'chart-note')
      .attr('x', 2).attr('y', y(R.floor.mmd2_max) + 14).style('font-size', '11.5px')
      .attr('fill', 'var(--anchor)').text('real against real: nothing below this');

    layer.append('path')
      .attr('d', d3.line().x((r) => x(r.steps)).y((r) => y(Math.max(Math.abs(r.mmd2), 1e-9)))(rows))
      .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2);
    layer.selectAll('circle').data(rows).join('circle')
      .attr('cx', (r) => x(r.steps)).attr('cy', (r) => y(Math.max(Math.abs(r.mmd2), 1e-9)))
      .attr('r', (r) => (r.steps === k ? 6 : 3.2)).attr('fill', 'var(--primary)');
    wrapLabel(title, `${st.mode === 'deterministic' ? 'deterministic' : 'ancestral'}: `
      + `inside the floor from ${st.mode === 'deterministic' ? R.enter_deterministic : R.enter_ancestral} steps on`, w - 8);

    caption.textContent = `${N_DRAW} draws made in this page, ${k} step${k === 1 ? '' : 's'} each, `
      + `${st.mode === 'deterministic' ? 'with the noise term off' : 'with the noise put back'}`;

    const row = rows.find((r) => r.steps === k);
    const hole = R.hole.find((r) => r.mode === st.mode);
    readout.innerHTML = `
      <table class="gen-table">
        <tr>
          <th>steps</th><th>distance</th><th>separable at</th>
          <th>in the hole here</th><th>inside the floor</th>
        </tr>
        <tr>
          <td><span class="value">${k}</span></td>
          <td><span class="value">${row.mmd2}</span></td>
          <td>${row.c2st}</td>
          <td>${inHole} of ${N_DRAW}</td>
          <td>${row.resolves ? 'no' : 'yes'}</td>
        </tr>
      </table>
      <div class="gen-note">
        The floor is the band: two pools of ${R.floor.n} real draws, split five ways, land
        ${R.floor.mmd2} apart on average and never more than ${R.floor.mmd2_max}, so anything
        under that is two samples of the same thing. The deterministic sampler is inside it from
        <span class="value">${R.enter_deterministic}</span> steps onwards and the ancestral one
        from <span class="value">${R.enter_ancestral}</span>, and "onwards" is the claim: it is
        checked against every larger budget, not against the first one that qualifies.
        The dashed circle is the ring's hole, and it is the one number here that does not improve.
        Measured properly, on ${hole.n.toLocaleString('en-US')} draws rather than the
        ${N_DRAW} drawn above, this sampler puts <span class="value">${hole.share}</span> of its
        mass in there against ${data.destroy.truth_hole} for the truth. The flows article's
        deepest flow put ${data.destroy.flow_hole}, and it had an excuse: a smooth invertible map
        cannot tear a gaussian open. This chain is not a smooth invertible map and does no better.
      </div>`;
  }

  slider.addEventListener('input', render);
  render();
  return { render };
}
