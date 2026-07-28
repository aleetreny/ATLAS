/* The neighbourhood graph, as a function of the only number Isomap asks for.
 *
 * Everything moves when the slider moves: the graph is rebuilt, all 124,750
 * shortest paths are recomputed, the table is double centred and two
 * eigenvectors come out of a block power iteration. About a third of a second,
 * so every value of k is cached after its first visit.
 *
 * Short circuit edges are drawn in the accent colour and everything else in
 * grey, because the whole argument of this section is that a handful of edges
 * out of thousands decide the answer, and a reader who cannot see which ones
 * has to take that on trust.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel, equalScales } from '../../assets/js/chart.js';
import { isomapAt } from './roll.js';

export function initKWidget(data, state, cache) {
  const G = data.graph;
  const chart = makeChart('#k-chart', {
    width: 780, height: 452, margin: { top: 84, right: 20, bottom: 60, left: 58 },
  });
  const { g, w, h } = chart;
  const viewButtons = document.querySelector('#k-view-buttons');
  const slider = document.querySelector('#k-slider');
  const label = document.querySelector('#k-label');
  const best = document.querySelector('#k-best');
  const readout = document.querySelector('#k-readout');
  const state2 = { view: 'both', k: data.graph.opens_at };

  const sVals = state.chart.map((c) => c[0]);
  const ramp = d3.scaleSequential(d3.interpolateTurbo)
    .domain([Math.min(...sVals), Math.max(...sVals)]);

  const btns = {};
  for (const [key, text] of [['both', 'the graph and the map'], ['curve', 'the whole sweep']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn' + (key === state2.view ? '' : ' ghost');
    b.textContent = text;
    b.addEventListener('click', () => {
      state2.view = key;
      render();
    });
    viewButtons.appendChild(b);
    btns[key] = b;
  }

  const body = g.append('g');
  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -66).attr('text-anchor', 'middle')
    .style('font-size', '12px').style('text-transform', 'none');
  const legend = g.append('g');

  /* The best k the slider can actually reach, swept at the slider's own
     resolution rather than at the data's, so it is both unbeatable and
     reachable. Same discipline as the dial in the directions article. */
  const reachable = G.rows.filter((r) => r.disparity !== undefined);
  const bestRow = reachable.reduce((a, b) => (b.disparity < a.disparity ? b : a));

  function drawGraph(panel, x0, width, R) {
    panel.selectAll('*').remove();
    const proj = state.proj;
    const { x, y } = equalScales(proj, width, h - 24, { pad: 1.06 });
    const shift = (v) => x0 + v;
    const shortSet = new Set(R.shorts.map(([i, j]) => `${i},${j}`));
    panel.selectAll('line.e').data(R.edges).join('line').attr('class', 'e')
      .attr('x1', (d) => shift(x(proj[d[0]][0]))).attr('y1', (d) => y(proj[d[0]][1]) + 20)
      .attr('x2', (d) => shift(x(proj[d[1]][0]))).attr('y2', (d) => y(proj[d[1]][1]) + 20)
      .attr('stroke', (d) => (shortSet.has(`${d[0]},${d[1]}`) ? 'var(--primary)' : 'var(--squidink)'))
      .attr('stroke-width', (d) => (shortSet.has(`${d[0]},${d[1]}`) ? 2.6 : 0.55))
      .attr('opacity', (d) => (shortSet.has(`${d[0]},${d[1]}`) ? 1 : 0.28));
    const order = proj.map((p, i) => i).sort((a, b) => proj[a][2] - proj[b][2]);
    panel.selectAll('circle').data(order).join('circle')
      .attr('cx', (i) => shift(x(proj[i][0]))).attr('cy', (i) => y(proj[i][1]) + 20)
      .attr('r', 2.6).attr('fill', (i) => ramp(state.chart[i][0])).attr('opacity', 0.9);
    panel.append('text').attr('x', x0 + width / 2).attr('y', 4)
      .attr('text-anchor', 'middle').attr('class', 'chart-annotation')
      .style('font-size', '11.5px').style('text-transform', 'none')
      .text(`the graph at k = ${R.k}`);
  }

  function drawMap(panel, x0, width, R) {
    panel.selectAll('*').remove();
    if (!R.coords) {
      panel.append('text').attr('x', x0 + width / 2).attr('y', h / 2)
        .attr('text-anchor', 'middle').attr('class', 'chart-annotation')
        .style('font-size', '12px').style('text-transform', 'none')
        .text(`${R.components} separate pieces: no map exists`);
      panel.append('text').attr('x', x0 + width / 2).attr('y', 4)
        .attr('text-anchor', 'middle').attr('class', 'chart-annotation')
        .style('font-size', '11.5px').style('text-transform', 'none')
        .text('the map');
      return;
    }
    const Z = R.aligned.coords;
    const ref = R.aligned.reference;
    const { x, y } = equalScales(ref.concat(Z), width, h - 24, { pad: 1.1 });
    panel.selectAll('circle.t').data(ref).join('circle').attr('class', 't')
      .attr('cx', (d) => x0 + x(d[0])).attr('cy', (d) => y(d[1]) + 20).attr('r', 4.6)
      .attr('fill', 'none').attr('stroke', 'var(--stone)').attr('stroke-width', 1);
    panel.selectAll('circle.z').data(Z).join('circle').attr('class', 'z')
      .attr('cx', (d) => x0 + x(d[0])).attr('cy', (d) => y(d[1]) + 20).attr('r', 2.8)
      .attr('fill', (d, i) => ramp(state.chart[i][0])).attr('opacity', 0.92);
    panel.append('text').attr('x', x0 + width / 2).attr('y', 4)
      .attr('text-anchor', 'middle').attr('class', 'chart-annotation')
      .style('font-size', '11.5px').style('text-transform', 'none')
      .text('the map, with the truth in grey');
  }

  function render() {
    const k = state2.k;
    if (!cache[k]) cache[k] = isomapAt(state, k);
    const R = cache[k];
    label.textContent = String(k);
    slider.value = String(k);
    for (const [key, b] of Object.entries(btns)) b.classList.toggle('ghost', key !== state2.view);
    body.selectAll('*').remove();
    legend.selectAll('*').remove();
    g.selectAll('g.axis, g.grid, text.axis-label').remove();

    if (state2.view === 'both') {
      const gap = 20;
      const half = (w - gap) / 2;
      drawGraph(body.append('g'), 0, half, R);
      drawMap(body.append('g'), half + gap, half, R);
      [['var(--primary)', `short circuit edges: ${R.shorts.length}`],
        ['var(--squidink)', `ordinary edges: ${R.edges.length - R.shorts.length}`]]
        .forEach(([c, t], i) => {
          legend.append('line').attr('x1', 4 + i * 220).attr('x2', 22 + i * 220)
            .attr('y1', -26).attr('y2', -26).attr('stroke', c)
            .attr('stroke-width', i === 0 ? 2.6 : 1);
          legend.append('text').attr('x', 28 + i * 220).attr('y', -22)
            .style('font-size', '11.5px').style('font-family', 'var(--font-mono)')
            .attr('fill', 'var(--squidink)').text(t);
        });
      wrapLabel(title, 'an edge is a short circuit when its two ends are near in the room and '
        + 'far along the sheet', w - 8);
    } else {
      const x = d3.scalePoint().domain(G.ks).range([10, w - 10]);
      const y = d3.scaleLinear().domain([0, 1]).range([h, 0]);
      drawGrid(g, x, y, w, h, { xTicks: 0, yTicks: 5 });
      drawAxes(g, x, y, w, h, { xTicks: G.ks.length, yTicks: 5 });
      axisLabels(g, w, h, { x: 'neighbours', y: 'score' });
      const rows = G.rows.filter((r) => r.disparity !== undefined);
      const line = (key) => d3.line().x((r) => x(r.k)).y((r) => y(r[key]));
      body.append('path').datum(rows).attr('fill', 'none')
        .attr('stroke', 'var(--primary)').attr('stroke-width', 2.4).attr('d', line('disparity'));
      body.append('path').datum(rows).attr('fill', 'none')
        .attr('stroke', 'var(--anchor)').attr('stroke-width', 2.4).attr('d', line('geodesic_corr'));
      const maxShort = d3.max(G.rows, (r) => r.short_circuits) || 1;
      body.selectAll('rect').data(G.rows).join('rect')
        .attr('x', (r) => x(r.k) - 4).attr('width', 8)
        .attr('y', (r) => y(r.short_circuits / maxShort))
        .attr('height', (r) => h - y(r.short_circuits / maxShort))
        .attr('fill', 'var(--stone)').attr('opacity', 0.75);
      const cur = G.rows.find((r) => r.k === k);
      body.append('line').attr('x1', x(k)).attr('x2', x(k)).attr('y1', 0).attr('y2', h)
        .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.4)
        .attr('stroke-dasharray', '4 3');
      if (cur && cur.disparity !== undefined) {
        body.append('circle').attr('cx', x(k)).attr('cy', y(cur.disparity)).attr('r', 5.5)
          .attr('fill', 'var(--primary)').attr('stroke', 'white').attr('stroke-width', 1.6);
        body.append('circle').attr('cx', x(k)).attr('cy', y(cur.geodesic_corr)).attr('r', 5.5)
          .attr('fill', 'var(--anchor)').attr('stroke', 'white').attr('stroke-width', 1.6);
      }
      [['var(--primary)', 'distance from the truth'],
        ['var(--anchor)', 'geodesic agreement'],
        ['var(--stone)', `short circuits (of ${maxShort.toLocaleString('en-US')})`]]
        .forEach(([c, t], i) => {
        legend.append('rect').attr('x', 4 + i * 245).attr('y', -30)
          .attr('width', 10).attr('height', 10).attr('fill', c);
        legend.append('text').attr('x', 20 + i * 245).attr('y', -21)
          .style('font-size', '11.5px').style('font-family', 'var(--font-mono)')
          .attr('fill', 'var(--squidink)').text(t);
      });
      wrapLabel(title, 'the whole sweep, with the slider\'s position marked', w - 8);
    }

    const ref = G.rows.find((r) => r.k === k);
    const broken = R.components > 1;
    readout.innerHTML = `
      <table class="man-table">
        <tr><th>neighbours</th><th>edges</th><th>short circuits</th><th>geodesic agreement</th><th>distance from the truth</th></tr>
        <tr>
          <td>${k}</td>
          <td>${R.edges.length.toLocaleString('en-US')}</td>
          <td><span class="value">${R.shorts.length}</span></td>
          <td>${broken ? 'the graph is in pieces' : `<span class="value">${R.geoCorr.toFixed(5)}</span>`}</td>
          <td>${broken ? 'no map' : `<span class="value">${R.disparity.toFixed(5)}</span>`}</td>
        </tr>
      </table>
      <div class="man-note">
        ${broken
    ? `At k = ${k} the graph falls into <span class="bold">${R.components} pieces</span>, and points `
      + `in different pieces have no path between them at all. Isomap has nothing to hand to `
      + `classical scaling. The graph first holds together at k = ${G.first_connected}.`
    : `${R.shorts.length === 0
      ? `Not one edge joins two turns of the roll, so every hop in this graph is a hop along the `
        + `sheet, and the ${(R.geoCorr * 100).toFixed(2)}% agreement with the true distances is what `
        + `that buys. The map is ${R.disparity.toFixed(5)} from the truth.`
      : `<span class="bold">${R.shorts.length}</span> of the ${R.edges.length.toLocaleString('en-US')} `
        + `edges jump between two turns of the roll, which is `
        + `${((R.shorts.length / R.edges.length) * 100).toFixed(2)}% of them. Those few edges are `
        + `shortcuts through empty space, so the shortest path between distant points now goes `
        + `through them, and the estimate of the distance along the sheet collapses: agreement `
        + `${R.geoCorr.toFixed(5)}, map ${R.disparity.toFixed(5)} from the truth against `
        + `<span class="bold">${bestRow.disparity}</span> at k = ${bestRow.k}.`}`}
        ${ref ? '' : ''}
      </div>`;
  }

  slider.addEventListener('input', () => {
    state2.k = +slider.value;
    render();
  });
  best.addEventListener('click', () => {
    state2.k = bestRow.k;
    render();
  });
  render();
  return { bestRow };
}
