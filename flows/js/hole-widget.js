/* The one thing a flow cannot do, drawn and measured.
 *
 * A flow is a smooth invertible map from a gaussian, and a smooth invertible
 * map cannot tear. The density this module is fitted against has a hole in the
 * middle of its ring, so the flow has to route mass around it or through it,
 * and it goes through: the picture shows the bridge, and the chart shows that
 * more layers do not remove it, they only make the map stretch harder.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';
import { makeCanvas, paintGray, canvasLabel } from '../../assets/js/imagery.js';
import { unpack } from './flowkit.js';

export function initHoleWidget(data, flow) {
  const P = data.picture;
  const n = P.n;
  const truth = unpack(P.truth_b64, n * n);
  const model = unpack(P.model_b64, n * n);
  const rows = data.topology.rows;
  const radii = data.topology.radii;
  const rr = radii[radii.length - 1];

  const ZOOM = Math.max(2, Math.round(340 / n));
  const cv = makeCanvas(document.querySelector('#hole-picture'), n * ZOOM, n * ZOOM);
  const chart = makeChart('#hole-chart', {
    width: 400, height: 340, margin: { top: 50, right: 26, bottom: 56, left: 76 },
  });
  const { g, w, h } = chart;
  chart.svg.style('max-width', '410px');
  const buttons = document.querySelector('#hole-buttons');
  const readout = document.querySelector('#hole-readout');
  const caption = document.querySelector('#hole-caption');
  const st = { view: 'model', metric: 'mass' };

  const btns = {};
  for (const [key, text] of [['truth', 'the truth'], ['model', 'what the flow learned'],
    ['samples', 'draws from the flow']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', () => { st.view = key; render(); });
    buttons.appendChild(b);
    btns[key] = b;
  }
  const flip = document.createElement('button');
  flip.className = 'atlas-btn ghost';
  flip.textContent = 'show how hard it stretches';
  flip.addEventListener('click', () => {
    st.metric = st.metric === 'mass' ? 'cond' : 'mass';
    flip.textContent = st.metric === 'mass' ? 'show how hard it stretches' : 'show the mass in the hole';
    render();
  });
  buttons.appendChild(flip);

  const peak = Math.sqrt(P.peak);
  const ink = (t) => [255 - 138 * t, 255 - 180 * t, 255 - 237 * t];
  const samples = flow.sample(1400, 909101);

  function paint() {
    const src = st.view === 'truth' ? truth : model;
    const vals = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        vals[(n - 1 - j) * n + i] = Math.min(1, Math.sqrt(src[i * n + j]) / peak);
      }
    }
    paintGray(cv.ctx, vals, n, n, { zoom: ZOOM, colour: ink });
    const span = P.hi - P.lo;
    const toPx = (p) => [((p[0] - P.lo) / span) * n * ZOOM, (1 - (p[1] - P.lo) / span) * n * ZOOM];
    if (st.view === 'samples') {
      cv.ctx.save();
      cv.ctx.fillStyle = 'rgba(35,47,62,0.8)';
      samples.forEach((p) => {
        const [cx, cy] = toPx(p);
        if (cx < 0 || cy < 0 || cx > n * ZOOM || cy > n * ZOOM) return;
        cv.ctx.beginPath();
        cv.ctx.arc(cx, cy, 1.3, 0, 2 * Math.PI);
        cv.ctx.fill();
      });
      cv.ctx.restore();
    }
    /* the circle every number in this section is about */
    cv.ctx.save();
    cv.ctx.strokeStyle = '#8a4b12';
    cv.ctx.lineWidth = 1.6;
    cv.ctx.setLineDash([5, 4]);
    const [ox, oy] = toPx([0, 0]);
    cv.ctx.beginPath();
    cv.ctx.arc(ox, oy, (rr / span) * n * ZOOM, 0, 2 * Math.PI);
    cv.ctx.stroke();
    cv.ctx.restore();
    canvasLabel(cv.ctx, `${P.lo}`, 4, n * ZOOM - 5, { align: 'start', size: 10 });
    canvasLabel(cv.ctx, `${P.hi}`, n * ZOOM - 4, n * ZOOM - 5, { align: 'end', size: 10 });
  }

  const layer = g.append('g');
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -32).attr('text-anchor', 'middle')
    .style('font-size', '12px');

  function render() {
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', k !== st.view));
    paint();
    layer.selectAll('*').remove();

    const depths = rows.map((r) => r.depth);
    const x = d3.scaleLinear().domain([0, Math.max(...depths) + 1]).range([0, w]);
    if (st.metric === 'mass') {
      const vals = rows.map((r) => r[`model_${rr}`]).concat([rows[0][`truth_${rr}`]]);
      const y = d3.scaleLog().domain([Math.min(...vals) * 0.4, Math.max(...vals) * 2.5])
        .range([h, 0]);
      const decades = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        .filter((v) => v >= y.domain()[0] && v <= y.domain()[1]);
      drawGrid(g, x, y, w, h, { xValues: depths, yValues: decades });
      drawAxes(g, x, y, w, h, { xValues: depths, yValues: decades, yFmt: d3.format('.0e') });
      axisLabels(g, w, h, { x: 'coupling layers', y: `mass inside radius ${rr}` });
      layer.append('path')
        .attr('d', d3.line().x((r) => x(r.depth)).y((r) => y(r[`model_${rr}`]))(rows))
        .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2);
      layer.selectAll('circle').data(rows).join('circle')
        .attr('cx', (r) => x(r.depth)).attr('cy', (r) => y(r[`model_${rr}`])).attr('r', 4.5)
        .attr('fill', 'var(--primary)');
      layer.append('line').attr('x1', 0).attr('x2', w)
        .attr('y1', y(rows[0][`truth_${rr}`])).attr('y2', y(rows[0][`truth_${rr}`]))
        .attr('stroke', 'var(--anchor)').attr('stroke-width', 1.6).attr('stroke-dasharray', '6 4');
      /* the truth's line sits at the bottom of the panel, so its label goes
         above it: below it lands on the axis ticks */
      layer.append('text').attr('class', 'chart-note')
        .attr('x', 2).attr('y', y(rows[0][`truth_${rr}`]) - 7).style('font-size', '11.5px')
        .attr('fill', 'var(--anchor)').text('what the truth puts there');
      wrapLabel(title, 'the hole, and how much the flow fills it', w - 8);
    } else {
      const vals = rows.map((r) => r.cond_max);
      const y = d3.scaleLinear().domain([0, Math.max(...vals) * 1.15]).range([h, 0]);
      drawGrid(g, x, y, w, h, { xValues: depths, yTicks: 5 });
      drawAxes(g, x, y, w, h, { xValues: depths, yTicks: 5 });
      axisLabels(g, w, h, { x: 'coupling layers', y: 'worst stretch of the map, at the data' });
      [['worst', (r) => r.cond_max, 'var(--primary)'],
        ['typical', (r) => r.cond_median, 'var(--anchor)']].forEach(([label, f, colour], si) => {
        layer.append('path')
          .attr('d', d3.line().x((r) => x(r.depth)).y((r) => y(f(r)))(rows))
          .attr('fill', 'none').attr('stroke', colour).attr('stroke-width', 2);
        layer.selectAll(`circle.s${si}`).data(rows).join('circle')
          .attr('class', `s${si}`)
          .attr('cx', (r) => x(r.depth)).attr('cy', (r) => y(f(r))).attr('r', 4)
          .attr('fill', colour);
        layer.append('text').attr('class', 'chart-note')
          .attr('x', w - 4).attr('y', 12 + si * 16).attr('text-anchor', 'end')
          .style('font-size', '11.5px').attr('fill', colour).text(label);
      });
      wrapLabel(title, 'what it does instead of tearing', w - 8);
    }

    caption.textContent = st.view === 'samples'
      ? `${samples.length} points pushed through the flow in this page, from the gaussian it starts from`
      : st.view === 'truth'
        ? `the closed form density, with the circle of radius ${rr} drawn on it`
        : `the flow's own density on the same square, same circle`;

    const deep = rows[rows.length - 1];
    const shallow = rows[0];
    readout.innerHTML = `
      <table class="gen-table">
        <tr>
          <th>layers</th><th>mass inside ${rr}</th><th>the truth puts</th>
          <th>distance on the ring</th><th>everywhere else</th><th>worst stretch</th>
        </tr>
        ${rows.map((r) => `<tr>
          <td>${r.depth}</td>
          <td><span class="value">${r[`model_${rr}`]}</span></td>
          <td>${r[`truth_${rr}`]}</td>
          <td>${r.kl_ring}</td>
          <td>${r.kl_rest}</td>
          <td>${r.cond_max}</td>
        </tr>`).join('')}
      </table>
      <div class="gen-note">
        The truth puts ${shallow[`truth_${rr}`]} of its mass inside that circle. The deepest flow
        measured puts ${deep[`model_${rr}`]}, which is
        <span class="value">${data.topology.ratio_deepest.toLocaleString('en-US')}</span> times as
        much. The column falls once, by a factor of
        ${data.topology.first_drop.toLocaleString('en-US')} between ${rows[0].depth} and
        ${rows[1].depth} layers, and then stops: the rest runs from ${data.topology.tail_min} to
        ${data.topology.tail_max} with no trend. This is not underfitting. A flow is a
        smooth invertible map applied to a gaussian, and a gaussian has no hole, so the image
        cannot have one either: the best the model can do is squeeze the bridge, and the last
        column is what that squeezing costs. The worst stretch of the map at the data goes from
        ${shallow.cond_max} at ${shallow.depth} layer${shallow.depth === 1 ? '' : 's'} to
        ${deep.cond_max} at ${deep.depth}, so the extra depth is being spent on the tear it is
        not allowed to make.
      </div>`;
  }

  render();
  return st;
}
