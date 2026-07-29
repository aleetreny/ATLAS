/* Bytes per vector against what those bytes buy.
 *
 * The second view is the trick every serving system uses and no comparison
 * table mentions: retrieve a wide candidate list with the cheap representation,
 * then recompute exact distances on that list alone. The cost of the second
 * stage is the width, in exact distances, and it is tiny next to the
 * collection, so the curve says how much of the loss can be bought back and for
 * how much.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initBytesWidget(data) {
  const B = data.bytes;
  const buttons = document.querySelector('#bytes-buttons');
  const readout = document.querySelector('#bytes-readout');
  const st = { view: 'bytes' };

  const chart = makeChart('#bytes-chart', {
    width: 640, height: 360, margin: { top: 54, right: 46, bottom: 58, left: 84 },
  });
  const { g, w, h } = chart;
  const layer = g.append('g');

  const btns = {};
  for (const [key, text] of [['bytes', 'what each size recovers'],
    ['rerank', 'buying the loss back']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', () => { st.view = key; render(); });
    buttons.appendChild(b);
    btns[key] = b;
  }

  function render() {
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', k !== st.view));
    layer.selectAll('*').remove();

    if (st.view === 'bytes') {
      const rows = B.rows.slice().sort((a, b) => a.bytes - b.bytes);
      const x = d3.scaleLog().domain([rows[0].bytes * 0.7, rows[rows.length - 1].bytes * 1.4])
        .range([0, w]);
      const y = d3.scaleLinear().domain([Math.min(...rows.map((r) => r.recall)) - 0.05, 1.02])
        .range([h, 0]);
      drawGrid(g, x, y, w, h, { xValues: rows.map((r) => r.bytes), yTicks: 5 });
      drawAxes(g, x, y, w, h, {
        xValues: rows.map((r) => r.bytes), yTicks: 5,
        xFmt: (v) => String(Math.round(v)), yFmt: d3.format('.2f'),
      });
      axisLabels(g, w, h, { x: 'bytes stored per vector', y: 'recall at ten' });
      layer.append('path').attr('d', d3.line().x((r) => x(r.bytes)).y((r) => y(r.recall))(rows))
        .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2.2);
      rows.forEach((r, i) => {
        layer.append('circle').attr('cx', x(r.bytes)).attr('cy', y(r.recall)).attr('r', 4.6)
          .attr('fill', 'var(--primary)');
        layer.append('text').attr('class', 'chart-note').attr('x', x(r.bytes))
          .attr('y', y(r.recall) + (i % 2 ? 18 : -12))
          .attr('text-anchor', i === rows.length - 1 ? 'end' : 'middle')
          .style('font-size', '10.5px').text(r.scheme);
      });
    } else {
      const rows = B.rerank;
      const x = d3.scaleLog().domain([rows[0].width * 0.8, rows[rows.length - 1].width * 1.3])
        .range([0, w]);
      const y = d3.scaleLinear().domain([Math.min(...rows.map((r) => r.recall)) - 0.05, 1.02])
        .range([h, 0]);
      drawGrid(g, x, y, w, h, { xValues: rows.map((r) => r.width), yTicks: 5 });
      drawAxes(g, x, y, w, h, {
        xValues: rows.map((r) => r.width), yTicks: 5,
        xFmt: (v) => String(v), yFmt: d3.format('.2f'),
      });
      axisLabels(g, w, h, {
        x: 'candidates rescored with exact distances', y: 'recall at ten',
      });
      layer.append('path').attr('d', d3.line().x((r) => x(r.width)).y((r) => y(r.recall))(rows))
        .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2.2);
      rows.forEach((r) => layer.append('circle').attr('cx', x(r.width)).attr('cy', y(r.recall))
        .attr('r', 4).attr('fill', 'var(--primary)'));
      layer.append('line').attr('x1', 0).attr('x2', w).attr('y1', y(1)).attr('y2', y(1))
        .attr('stroke', 'var(--smile)').attr('stroke-width', 1.4).attr('stroke-dasharray', '6 4');
      layer.append('text').attr('class', 'chart-note').attr('x', w - 2).attr('y', y(1) + 14)
        .attr('text-anchor', 'end').style('font-size', '11px').attr('fill', 'var(--smile)')
        .text('exhaustive search');
    }

    const rows = B.rows;
    const float = rows[0];
    const smallest = rows.reduce((a, b) => (b.bytes < a.bytes ? b : a));
    const rer = B.rerank;
    const enough = rer.find((r) => r.recall >= 0.95) || rer[rer.length - 1];
    const tbl = rows.map((r) => `<tr><td>${r.scheme}</td><td>${Math.round(r.bytes)}</td>`
      + `<td>${(float.bytes / r.bytes).toFixed(1)} times less</td>`
      + `<td>${r.recall.toFixed(4)}</td><td>${r.quality.toFixed(4)}</td></tr>`).join('');
    readout.innerHTML = `<table class="gen-table"><thead><tr><th>stored as</th>`
      + `<th>bytes per vector</th><th>against float32</th><th>recall at ten</th>`
      + `<th>right category share</th></tr></thead><tbody>${tbl}</tbody></table>`
      + `<div class="gen-note">The smallest representation here is `
      + `${Math.round(smallest.bytes)} bytes per vector against ${Math.round(float.bytes)} for `
      + `full precision, ${(float.bytes / smallest.bytes).toFixed(0)} times less, at `
      + `${smallest.recall.toFixed(4)} recall. Most of that loss is recoverable and the way `
      + `to recover it is the second view: retrieve a wide list cheaply, then recompute exact `
      + `distances on that list alone. With ${enough.width} candidates rescored the recall is `
      + `${enough.recall.toFixed(4)}, and the second stage costs ${enough.width} exact `
      + `distances, which is ${(enough.width / B.n * 100).toFixed(2)}% of the collection. `
      + `That is the shape of every serving system in production: a cheap representation for `
      + `the shortlist and an expensive one for the ten that get returned.</div>`;
  }

  render();
  return { render, state: st };
}
