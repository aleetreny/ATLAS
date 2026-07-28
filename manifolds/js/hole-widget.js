/* The same roll with a rectangle cut out of it, which is the case Isomap's
 * argument does not cover.
 *
 * Classical scaling assumes the distances can be realised by points in a flat
 * space. Graph distances on a sheet with a hole cannot: every path between the
 * two sides has to go round, so the table says those pairs are further apart
 * than any flat arrangement can make them while keeping everything else right.
 * The embedding has to compromise, and the compromise is measurable.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel, equalScales } from '../../assets/js/chart.js';
import { pairwise, procrustes, toRows } from '../../assets/js/embed.js';
import { isomapAt, neighbourOrder, project3 } from './roll.js';

export function holeState(data) {
  const points = toRows(data.hole.points);
  const chart = toRows(data.hole.chart);
  const D = pairwise(points);
  return {
    points,
    chart,
    D,
    GEO: pairwise(chart),
    order: neighbourOrder(D),
    short: data.meta.short_factor,
    proj: project3(points),
  };
}

export function initHoleWidget(data, intact, holed, cache) {
  const chart = makeChart('#hole-chart', {
    width: 780, height: 300, margin: { top: 78, right: 22, bottom: 30, left: 40 },
  });
  const { g, w, h } = chart;
  const buttons = document.querySelector('#hole-buttons');
  const readout = document.querySelector('#hole-readout');
  const st = { key: 'holed' };
  const K = data.meta.k_show;

  const btns = {};
  for (const [key, text] of [['intact', 'the intact roll'], ['holed', 'with a hole cut out']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn' + (key === st.key ? '' : ' ghost');
    b.textContent = text;
    b.addEventListener('click', () => {
      st.key = key;
      render();
    });
    buttons.appendChild(b);
    btns[key] = b;
  }

  const body = g.append('g');
  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -60).attr('text-anchor', 'middle')
    .style('font-size', '12px').style('text-transform', 'none');

  function ramp(s) {
    const all = (st.key === 'holed' ? holed : intact).chart.map((c) => c[0]);
    return d3.scaleSequential(d3.interpolateTurbo)
      .domain([Math.min(...all), Math.max(...all)])(s);
  }

  function render() {
    const S = st.key === 'holed' ? holed : intact;
    if (!cache[st.key]) cache[st.key] = isomapAt(S, K);
    const R = cache[st.key];
    for (const [k, b] of Object.entries(btns)) b.classList.toggle('ghost', k !== st.key);
    body.selectAll('*').remove();
    g.selectAll('g.axis, g.grid, text.axis-label').remove();

    const gap = 20;
    const half = (w - gap) / 2;
    /* left: the sheet as it really is, unrolled, so the hole is visible as a
       hole rather than as a suspicious gap in a 3-D tangle */
    const left = body.append('g');
    const { x: cx, y: cy } = equalScales(S.chart, half, h - 22, { pad: 1.06 });
    left.selectAll('circle').data(S.chart).join('circle')
      .attr('cx', (d) => cx(d[0])).attr('cy', (d) => cy(d[1]) + 18).attr('r', 2.6)
      .attr('fill', (d) => ramp(d[0])).attr('opacity', 0.9);
    left.append('text').attr('x', half / 2).attr('y', 2).attr('text-anchor', 'middle')
      .attr('class', 'chart-annotation').style('font-size', '11.5px')
      .style('text-transform', 'none').text('the sheet, as it really is');

    const right = body.append('g');
    const A = procrustes(S.chart, R.coords);
    const { x, y } = equalScales(A.reference.concat(A.coords), half, h - 22, { pad: 1.1 });
    right.selectAll('circle.t').data(A.reference).join('circle').attr('class', 't')
      .attr('cx', (d) => half + gap + x(d[0])).attr('cy', (d) => y(d[1]) + 18).attr('r', 4.4)
      .attr('fill', 'none').attr('stroke', 'var(--stone)').attr('stroke-width', 1);
    right.selectAll('circle.z').data(A.coords).join('circle').attr('class', 'z')
      .attr('cx', (d) => half + gap + x(d[0])).attr('cy', (d) => y(d[1]) + 18).attr('r', 2.6)
      .attr('fill', (d, i) => ramp(S.chart[i][0])).attr('opacity', 0.92);
    right.append('text').attr('x', half + gap + half / 2).attr('y', 2)
      .attr('text-anchor', 'middle').attr('class', 'chart-annotation')
      .style('font-size', '11.5px').style('text-transform', 'none')
      .text(`what Isomap makes of it at k = ${K}`);

    wrapLabel(title, st.key === 'holed'
      ? 'every path between the two sides of the hole has to go round it, and a flat map has no '
        + 'way to make that true'
      : 'with nothing in the way, the map and the sheet are the same picture', w - 8);

    const H = data.hole;
    const dInt = cache.intact.disparity;
    const dHole = cache.holed.disparity;
    readout.innerHTML = `
      <table class="man-table">
        <tr><th>sheet</th><th>points</th><th>distance from the truth</th><th>path over flat distance</th></tr>
        <tr${st.key === 'intact' ? ' style="font-weight:700"' : ''}>
          <td>intact</td><td>${intact.points.length}</td>
          <td><span class="value">${dInt.toFixed(5)}</span></td>
          <td>${H.ratio_intact}</td></tr>
        <tr${st.key === 'holed' ? ' style="font-weight:700"' : ''}>
          <td>with a hole</td><td>${holed.points.length}</td>
          <td><span class="value">${dHole.toFixed(5)}</span></td>
          <td>${H.ratio_hole}</td></tr>
      </table>
      <div class="man-note">
        A graph path is a chain of straight hops across a curved sheet, so it always undershoots the
        real arc a little: on the intact roll it comes to ${H.ratio_intact} times the flat distance,
        and that overhead is the same for any sheet. With the hole it is ${H.ratio_hole}, a detour of
        <span class="bold">${H.detour}</span> that the hole and nothing else added, and
        ${H.crossing_pairs.toLocaleString('en-US')} of the
        ${H.total_pairs.toLocaleString('en-US')} pairs have to walk more than ${H.threshold} times
        the flat distance, which the intact roll never asks of anybody. The map pays for it:
        ${dHole.toFixed(5)} from the true chart against ${dInt.toFixed(5)} on the intact roll,
        while trustworthiness stays at ${H.trustworthiness}. The local structure is fine.
        The global shape is a compromise, because the shape the algorithm assumes exists does not.
      </div>`;
  }

  render();
}
