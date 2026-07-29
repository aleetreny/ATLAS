/* The field the network is trying to learn, beside the one it learned.
 *
 * This is the widget the article exists for. On images nobody can draw the left
 * hand panel, because writing down the true score means writing down the data
 * density and that is the whole problem. Here the forward process is closed
 * form, so the page can compute the true field at any point and any noise
 * level, put the network's answer next to it, and subtract them.
 *
 * The picture is the density q(x_t) with arrows on it; the chart is the error
 * against the noise level, weighted by where the density actually is, because
 * an error where nothing lives is not an error anyone pays for.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';
import { makeCanvas, paintGray, canvasLabel } from '../../assets/js/imagery.js';

export function initScoreWidget(data, kit) {
  const S = data.score;
  const levels = data.net.page_ts;
  const slider = document.querySelector('#score-slider');
  const valueTag = document.querySelector('#score-value');
  const caption = document.querySelector('#score-caption');
  const readout = document.querySelector('#score-readout');
  const buttons = document.querySelector('#score-buttons');
  /* opens at the second level, where the two fields visibly differ and the
     density still has structure: the first is the worst case and the ones
     after it are where the network is already right */
  const st = { view: 'both', level: 1 };

  slider.min = 0;
  slider.max = levels.length - 1;
  slider.value = st.level;

  const btns = {};
  for (const [key, text] of [['both', 'both fields'], ['true', 'the true one'],
    ['learned', 'what the network learned']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', () => { st.view = key; render(); });
    buttons.appendChild(b);
    btns[key] = b;
  }

  const NPIX = 132;
  const HALF = 4.6;
  const ZOOM = Math.max(2, Math.round(340 / NPIX));
  const cv = makeCanvas(document.querySelector('#score-canvas'), NPIX * ZOOM, NPIX * ZOOM);
  const chart = makeChart('#score-chart', {
    width: 400, height: 330, margin: { top: 50, right: 26, bottom: 56, left: 74 },
  });
  const { g, w, h } = chart;
  /* the width is the column's, set in the page's stylesheet: see .chart-half */
  const layer = g.append('g');
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -32).attr('text-anchor', 'middle')
    .style('font-size', '12px');

  const ink = (t) => [255 - 152 * t, 255 - 186 * t, 255 - 102 * t];

  function paint(level) {
    /* the density, computed here in closed form rather than shipped as a
       picture: what is drawn is the function, at whatever resolution the canvas
       happens to be */
    const vals = new Float64Array(NPIX * NPIX);
    let peak = 0;
    for (let i = 0; i < NPIX; i++) {
      for (let j = 0; j < NPIX; j++) {
        const x = -HALF + ((i + 0.5) / NPIX) * 2 * HALF;
        const y = -HALF + ((j + 0.5) / NPIX) * 2 * HALF;
        const q = kit.truthAt(level, x, y).q;
        vals[(NPIX - 1 - j) * NPIX + i] = q;
        if (q > peak) peak = q;
      }
    }
    const root = Math.sqrt(peak);
    for (let k = 0; k < vals.length; k++) vals[k] = Math.min(1, Math.sqrt(vals[k]) / root);
    paintGray(cv.ctx, vals, NPIX, NPIX, { zoom: ZOOM, colour: ink });

    const px = (x) => ((x + HALF) / (2 * HALF)) * NPIX * ZOOM;
    const py = (y) => (1 - (y + HALF) / (2 * HALF)) * NPIX * ZOOM;
    const G = 11;
    /* Two arrows from the same point, and the first version drew them the same
       width: wherever the network is right (which is most of the plane) the
       learned arrow covered the true one exactly and the picture looked like a
       single field. The true field is drawn WIDE and the learned one THIN on
       top of it, so agreement reads as a thin line inside a thick one and
       disagreement is the only thing that separates. */
    const arrow = (x0, y0, sx, sy, colour, width, dot) => {
      const n = Math.hypot(sx, sy);
      if (n < 1e-9) return;
      const len = Math.min(18, 7.5 * n);
      const ux = (sx / n) * len;
      const uy = (sy / n) * len;
      const ax = px(x0);
      const ay = py(y0);
      cv.ctx.strokeStyle = colour;
      cv.ctx.fillStyle = colour;
      cv.ctx.lineWidth = width;
      cv.ctx.lineCap = 'round';
      cv.ctx.beginPath();
      cv.ctx.moveTo(ax, ay);
      cv.ctx.lineTo(ax + ux, ay - uy);
      cv.ctx.stroke();
      if (dot) {
        cv.ctx.beginPath();
        cv.ctx.arc(ax + ux, ay - uy, dot, 0, Math.PI * 2);
        cv.ctx.fill();
      }
    };
    for (let i = 0; i < G; i++) {
      for (let j = 0; j < G; j++) {
        const x = -HALF * 0.86 + ((i + 0.5) / G) * 2 * HALF * 0.86;
        const y = -HALF * 0.86 + ((j + 0.5) / G) * 2 * HALF * 0.86;
        if (st.view !== 'learned') {
          const T = kit.truthAt(level, x, y);
          arrow(x, y, T.sx, T.sy, 'rgba(103,69,153,0.55)', 5.0, 0);
        }
        if (st.view !== 'true') {
          const M = kit.learnedAt(level, x, y);
          arrow(x, y, M.sx, M.sy, 'rgba(154,52,18,0.95)', 1.5, 1.7);
        }
      }
    }
    canvasLabel(cv.ctx, `${-HALF}`, 4, NPIX * ZOOM - 5, { align: 'start', size: 10 });
    canvasLabel(cv.ctx, `${HALF}`, NPIX * ZOOM - 4, NPIX * ZOOM - 5, { align: 'end', size: 10 });
  }

  function render() {
    st.level = +slider.value;
    const t = levels[st.level];
    valueTag.textContent = `t = ${t}`;
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', k !== st.view));
    paint(st.level);
    layer.selectAll('*').remove();

    const rows = S.rows;
    const x = d3.scaleLinear().domain([0, data.meta.steps]).range([0, w]);
    const y = d3.scaleLog()
      .domain([Math.max(1e-3, d3.min(rows, (r) => r.rel) * 0.6), d3.max(rows, (r) => r.rel) * 1.6])
      .range([h, 0]);
    const decades = [1e-3, 1e-2, 1e-1, 1, 10]
      .filter((v) => v >= y.domain()[0] && v <= y.domain()[1]);
    drawGrid(g, x, y, w, h, { xTicks: 5, yValues: decades });
    drawAxes(g, x, y, w, h, { xTicks: 5, yValues: decades, yFmt: d3.format('.0e') });
    axisLabels(g, w, h, { x: 'noise level t', y: 'error against the truth' });

    layer.append('path')
      .attr('d', d3.line().x((r) => x(r.t)).y((r) => y(r.rel))(rows))
      .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2);
    layer.selectAll('circle').data(rows).join('circle')
      .attr('cx', (r) => x(r.t)).attr('cy', (r) => y(r.rel))
      .attr('r', (r) => (r.t === t ? 6 : 3.2)).attr('fill', 'var(--primary)');
    /* the level the picture is showing, marked on the curve */
    layer.append('line').attr('x1', x(t)).attr('x2', x(t)).attr('y1', 0).attr('y2', h)
      .attr('stroke', 'var(--anchor)').attr('stroke-width', 1.2).attr('stroke-dasharray', '4 3')
      .attr('opacity', 0.8);
    wrapLabel(title, 'the error is worst where the data still is, not where the noise is', w - 8);

    caption.textContent = st.view === 'both'
      ? `q(x_t) at t = ${t}, with the true score in the article's colour and the network's in orange`
      : st.view === 'true'
        ? `q(x_t) at t = ${t}, with the exact score, computed in this page from the closed form`
        : `q(x_t) at t = ${t}, with the score the network learned, from its float16 weights`;

    const row = rows.reduce((a, b) => (Math.abs(b.t - t) < Math.abs(a.t - t) ? b : a));
    const worst = rows.reduce((a, b) => (b.rel > a.rel ? b : a));
    const best = rows.reduce((a, b) => (b.rel < a.rel ? b : a));
    readout.innerHTML = `
      <table class="gen-table">
        <tr>
          <th>t</th><th>alignment</th><th>error</th>
          <th>true length</th><th>learned length</th>
        </tr>
        <tr>
          <td><span class="value">${row.t}</span></td>
          <td>${row.cos}</td>
          <td><span class="value">${row.rel}</span></td>
          <td>${row.true_norm}</td>
          <td>${row.learned_norm}</td>
        </tr>
      </table>
      <div class="gen-note">
        Both numbers are averages over the plane weighted by <i>q(x_t) itself</i>, so an error
        somewhere the density never goes does not count. Alignment is the cosine between the two
        fields, one being perfect. The error is the length of the difference divided by the length
        of the truth, so ${worst.rel} at t = ${worst.t} means the network's arrow is wrong by
        ${worst.rel > 1 ? 'more than the whole' : 'a fraction of the'} true arrow.
        It is worst at t = ${worst.t} and best at t = ${best.t}, and that ordering is the
        article's point: the noisy end is nearly free, because there the answer is almost
        <i>x</i> pointing at the origin, and the clean end is where all the structure is.
        The last row of the schedule is a check rather than a result: with the data gone,
        q(x_t) is a standard normal, whose score is minus x, so the true field must have mean
        length ${data.guards.end_score_target} there. It has ${data.guards.end_score_norm}.
      </div>`;
  }

  slider.addEventListener('input', render);
  render();
  return { render };
}
