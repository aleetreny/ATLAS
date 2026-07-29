/* The trajectories, drawn, which is the only honest way to say "curved".
 *
 * Eight numbers do not fit on a page, so they are projected onto their own two
 * strongest directions by a matrix the generator computed and shipped. That is
 * a shadow, and a shadow of a straight line is straight while a shadow of a
 * curve can be anything: so the deviation NUMBER on this page is measured in
 * all eight dimensions, in the generator, and the drawing is here to make the
 * number believable rather than to be the evidence itself.
 *
 * The thin line is the hundred step walk, which is as close to the model's real
 * trajectory as this article ever gets. The thick line with the dots is what
 * the reader's step budget actually walks. The gap between where the two of
 * them stop is the whole subject.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';

const ARMS = [['flow', 'rectified flow'], ['reflow', 'after reflow'],
  ['student', 'distilled student']];

export function initPathWidget(data) {
  const P = data.paths;
  const C = data.cloud;
  const B = data.meta.budgets;
  const slider = document.querySelector('#path-slider');
  const valueTag = document.querySelector('#path-value');
  const buttons = document.querySelector('#path-buttons');
  const readout = document.querySelector('#path-readout');
  const st = { arm: 'flow', budget: 0 };

  slider.min = 0;
  slider.max = B.length - 1;
  slider.value = 0;

  const btns = {};
  ARMS.forEach(([key, text]) => {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', () => { st.arm = key; render(); });
    buttons.appendChild(b);
    btns[key] = b;
  });

  const chart = makeChart('#path-chart', {
    width: 620, height: 470, margin: { top: 46, right: 26, bottom: 54, left: 58 },
  });
  const { g, w, h } = chart;
  const cloudLayer = g.append('g');
  const layer = g.append('g');
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -28).attr('text-anchor', 'middle')
    .style('font-size', '12px');

  /* one domain for every arm and every budget, so switching does not move the
     picture underneath the reader */
  let lo = -C.limit;
  let hi = C.limit;
  Object.values(P).forEach((entry) => {
    const sets = [];
    if (entry.ref) sets.push(...entry.ref);
    if (entry.coarse) Object.values(entry.coarse).forEach((v) => sets.push(...v));
    if (entry.start) sets.push(entry.start, entry.jump);
    sets.forEach((pts) => pts.forEach((p) => {
      lo = Math.min(lo, p[0], p[1]);
      hi = Math.max(hi, p[0], p[1]);
    }));
  });
  const pad = 0.06 * (hi - lo);
  const x = d3.scaleLinear().domain([lo - pad, hi + pad]).range([0, w]);
  const y = d3.scaleLinear().domain([lo - pad, hi + pad]).range([h, 0]);

  /* the real codes, as a faint field, so the paths have somewhere to be going */
  const cell = (2 * C.limit) / C.bins;
  const maxCount = Math.max(...C.counts.map((row) => Math.max(...row)));
  C.counts.forEach((row, j) => row.forEach((v, i) => {
    if (!v) return;
    cloudLayer.append('rect')
      .attr('x', x(-C.limit + i * cell)).attr('y', y(-C.limit + (j + 1) * cell))
      .attr('width', Math.abs(x(cell) - x(0)) + 0.6)
      .attr('height', Math.abs(y(cell) - y(0)) + 0.6)
      .attr('fill', 'var(--squidink)')
      .attr('opacity', 0.03 + 0.32 * Math.sqrt(v / maxCount));
  }));

  const line = d3.line().x((p) => x(p[0])).y((p) => y(p[1]));

  function render() {
    st.budget = +slider.value;
    const steps = B[st.budget];
    valueTag.textContent = `${steps} step${steps === 1 ? '' : 's'}`;
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', k !== st.arm));
    slider.disabled = st.arm === 'student';
    layer.selectAll('*').remove();
    drawGrid(g, x, y, w, h, { xTicks: 6, yTicks: 6 });
    drawAxes(g, x, y, w, h, { xTicks: 6, yTicks: 6 });
    axisLabels(g, w, h, { x: 'strongest direction of the latent', y: 'second strongest' });

    /* A legend, because without one the reader has to guess which of two line
       weights is the shortcut, and the first draft of this widget was exactly
       that guess: ten thick coloured lines all diving into the middle, which
       reads as "the coloured one is the good one" when it is the opposite.
       Drawn in the top left, which the cloud never reaches. */
    const LEG = st.arm === 'student'
      ? [['start', 'ring'], ['one evaluation, straight there', 'thick']]
      : [['where it starts', 'ring'],
        ['a hundred steps, the reference', 'thin'],
        [`${steps} step${steps === 1 ? '' : 's'}, what you asked for`, 'thick']];
    LEG.forEach(([text, kind], i) => {
      const ly = 12 + i * 17;
      if (kind === 'ring') {
        layer.append('circle').attr('cx', 10).attr('cy', ly - 4).attr('r', 2.6)
          .attr('fill', 'var(--stone)').attr('stroke', 'var(--squidink)')
          .attr('stroke-width', 0.9);
      } else {
        layer.append('line').attr('x1', 2).attr('x2', 18).attr('y1', ly - 4).attr('y2', ly - 4)
          .attr('stroke', kind === 'thin' ? 'var(--squidink)' : 'var(--primary)')
          .attr('stroke-width', kind === 'thin' ? 1.1 : 2.1)
          .attr('opacity', kind === 'thin' ? 0.5 : 0.9);
        /* the swatch carries the END MARKER too, because the chart has three
           kinds of round mark on it and the hollow ring is the one a reader is
           most likely to confuse with a start */
        if (kind === 'thin') {
          layer.append('circle').attr('cx', 19).attr('cy', ly - 4).attr('r', 4.6)
            .attr('fill', 'none').attr('stroke', 'var(--squidink)').attr('stroke-width', 1.4);
        } else {
          layer.append('circle').attr('cx', 19).attr('cy', ly - 4).attr('r', 4.2)
            .attr('fill', 'var(--primary)');
        }
      }
      layer.append('text').attr('class', 'chart-note')
        .attr('x', 30).attr('y', ly).attr('text-anchor', 'start')
        .style('font-size', '11px')
        .attr('fill', kind === 'thick' ? 'var(--primary)' : 'var(--squidink)')
        .text(text);
    });

    let note;
    if (st.arm === 'student') {
      /* no trajectory to draw: that is the point, so it is drawn as the one
         jump it makes, from the noise it was handed to the answer */
      P.student.start.forEach((s, i) => {
        const e = P.student.jump[i];
        layer.append('line').attr('x1', x(s[0])).attr('y1', y(s[1]))
          .attr('x2', x(e[0])).attr('y2', y(e[1]))
          .attr('stroke', 'var(--primary)').attr('stroke-width', 2)
          .attr('opacity', 0.8);
        layer.append('circle').attr('cx', x(s[0])).attr('cy', y(s[1])).attr('r', 3)
          .attr('fill', 'none').attr('stroke', 'var(--squidink)').attr('stroke-width', 1.2);
        layer.append('circle').attr('cx', x(e[0])).attr('cy', y(e[1])).attr('r', 4.2)
          .attr('fill', 'var(--primary)');
      });
      note = 'the student has no trajectory: one evaluation, start to finish';
    } else {
      const entry = P[st.arm];
      const coarse = entry.coarse[String(steps)];
      entry.ref.forEach((pts, i) => {
        layer.append('path').attr('d', line(pts))
          .attr('fill', 'none').attr('stroke', 'var(--squidink)')
          .attr('stroke-width', 1.1).attr('opacity', 0.5);
        /* where the fine walk stops, as a hollow ring, so the coarse walk's
           filled dot can be compared against it without a legend */
        const last = pts[pts.length - 1];
        layer.append('circle').attr('cx', x(last[0])).attr('cy', y(last[1])).attr('r', 4.6)
          .attr('fill', 'none').attr('stroke', 'var(--squidink)').attr('stroke-width', 1.4);
        layer.append('circle').attr('cx', x(pts[0][0])).attr('cy', y(pts[0][1])).attr('r', 2.6)
          .attr('fill', 'var(--stone)').attr('stroke', 'var(--squidink)')
          .attr('stroke-width', 0.8);
      });
      coarse.forEach((pts) => {
        layer.append('path').attr('d', line(pts))
          .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2.1)
          .attr('opacity', 0.9);
        pts.slice(1).forEach((p) => {
          layer.append('circle').attr('cx', x(p[0])).attr('cy', y(p[1]))
            .attr('r', steps > 16 ? 1.3 : 2.6).attr('fill', 'var(--primary)');
        });
        const end = pts[pts.length - 1];
        layer.append('circle').attr('cx', x(end[0])).attr('cy', y(end[1])).attr('r', 4.2)
          .attr('fill', 'var(--primary)');
      });
      const gap = data.gaps.find(
        (r) => r.arm === (st.arm === 'flow' ? 'rectified flow' : 'after reflow')
          && r.steps === steps);
      note = steps >= 100
        ? 'at a hundred steps the two are the same walk, which is what makes it the reference'
        : `${steps} step${steps === 1 ? '' : 's'} lands `
          + `${(gap.gap * 100).toFixed(1)}% of the trip away from where a hundred do`;
    }
    wrapLabel(title, note, w - 8);

    const armName = st.arm === 'flow' ? 'rectified flow'
      : (st.arm === 'reflow' ? 'after reflow' : 'distilled student');
    const rows = st.arm === 'student'
      ? [{ steps: 1, gap: 0 }]
      : data.gaps.filter((r) => r.arm === armName);
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>steps</th>${rows.map((r) => `<th>${r.steps}</th>`).join('')}</tr>
        <tr><td>away from the fine walk</td>${rows.map((r) => `<td>${
      st.arm === 'student' ? 'n/a' : `<span class="${r.steps === B[st.budget] ? 'value' : ''}">`
        + `${(r.gap * 100).toFixed(1)}%</span>`}</td>`).join('')}</tr>
      </table>
      <div class="gen-note">
        ${st.arm === 'student'
    ? `There is no row here because there is no chain to shorten. The student was trained to
       name the endpoint of a teacher trajectory from anywhere on it, so it is asked once and
       the answer is the sample. What it costs is on the chart below, in the same units as
       everything else.`
    : `Every number is a distance between two endpoints in the eight number latent, over the
       length of the whole trip, averaged over 250 starting points. It is not a picture and it
       is not a judgement: it is how far the shortcut moved the answer, before any of it
       reaches a decoder. The paths above are a shadow of that in two dimensions, and the
       reason the deviation is measured in eight rather than read off this drawing.`}
      </div>`;
  }

  slider.addEventListener('input', render);
  render();
  return { render };
}
