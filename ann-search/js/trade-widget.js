/* The one chart the subject is about: how much you get for how much you pay.
 *
 * Two views, and the second is the one nobody draws. Recall against cost is the
 * standard picture, and it treats a missing neighbour as a loss. But nobody
 * wants neighbours; they want whatever the neighbours are for. With the Reuters
 * categories as an answer key, the same runs can be scored on how many of the
 * ten returned documents share the query's category, and the two curves have
 * very different shapes.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initTradeWidget(data) {
  const I = data.indices;
  const buttons = document.querySelector('#trade-buttons');
  const readout = document.querySelector('#trade-readout');
  const st = { view: 'recall' };

  const chart = makeChart('#trade-chart', {
    width: 640, height: 390, margin: { top: 58, right: 40, bottom: 58, left: 82 },
  });
  const { g, w, h } = chart;
  const layer = g.append('g');

  const btns = {};
  for (const [key, text] of [['recall', 'recall of the true neighbours'],
    ['quality', 'how many of the ten are the right category']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', () => { st.view = key; render(); });
    buttons.appendChild(b);
    btns[key] = b;
  }

  const SERIES = [
    ['hnsw', 'navigable graph', 'var(--primary)', (r) => r.ef],
    ['ivf', 'inverted file', 'var(--anchor)', (r) => r.nprobe],
    ['pq', 'product quantiser', 'var(--cosmos)', (r) => r.m],
  ];

  function render() {
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', k !== st.view));
    layer.selectAll('*').remove();
    const key = st.view === 'recall' ? 'recall' : 'quality';

    const costs = SERIES.flatMap(([s]) => I[s].map((r) => r.cost)).concat([I.brute_cost]);
    const x = d3.scaleLog().domain([d3.min(costs) * 0.7, I.brute_cost * 1.6]).range([0, w]);
    const vals = SERIES.flatMap(([s]) => I[s].map((r) => r[key]))
      .concat([st.view === 'recall' ? 1 : I.exact_quality]);
    const y = d3.scaleLinear().domain([d3.min(vals) - 0.04, d3.max(vals) + 0.03]).range([h, 0]);
    drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
    drawAxes(g, x, y, w, h, {
      xTicks: 5, yTicks: 5, yFmt: d3.format('.2f'),
      xFmt: (v) => (v >= 1000 ? `${(v / 1000).toFixed(1)}k` : String(Math.round(v))),
    });
    axisLabels(g, w, h, {
      x: 'distances computed per query',
      y: st.view === 'recall' ? 'recall at ten' : 'share of the ten in the right category',
    });

    const exactY = st.view === 'recall' ? 1 : I.exact_quality;
    layer.append('line').attr('x1', 0).attr('x2', w).attr('y1', y(exactY)).attr('y2', y(exactY))
      .attr('stroke', 'var(--smile)').attr('stroke-width', 1.4).attr('stroke-dasharray', '6 4');
    layer.append('circle').attr('cx', x(I.brute_cost)).attr('cy', y(exactY)).attr('r', 5)
      .attr('fill', 'var(--smile)');
    layer.append('text').attr('class', 'chart-note').attr('x', w - 2)
      .attr('y', y(exactY) + 15).attr('text-anchor', 'end').style('font-size', '11px')
      .attr('fill', 'var(--smile)')
      .text(`exhaustive search: ${Math.round(I.brute_cost)} distances, `
        + `${exactY.toFixed(3)}`);

    SERIES.forEach(([s, name, colour, knob], i) => {
      const rows = I[s].slice().sort((a, b) => a.cost - b.cost);
      layer.append('path')
        .attr('d', d3.line().x((r) => x(r.cost)).y((r) => y(r[key]))(rows))
        .attr('fill', 'none').attr('stroke', colour).attr('stroke-width', 2.2);
      rows.forEach((r) => {
        layer.append('circle').attr('cx', x(r.cost)).attr('cy', y(r[key])).attr('r', 4)
          .attr('fill', colour);
      });
      const lab = rows[rows.length - 1];
      layer.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -46 + i * 15)
        .style('font-size', '11px').attr('fill', colour)
        .text(`${name}, dial from ${knob(rows[0])} to ${knob(lab)}`);
    });

    const target = 0.9;
    const rows = SERIES.map(([s, name]) => {
      const sorted = I[s].slice().sort((a, b) => a.cost - b.cost);
      const hit = sorted.find((r) => r.recall >= target);
      const best = sorted[sorted.length - 1];
      return `<tr><td>${name}</td>`
        + `<td>${hit ? Math.round(hit.cost) : 'never'}</td>`
        + `<td>${hit ? `${(I.brute_cost / hit.cost).toFixed(1)} times less` : ''}</td>`
        + `<td>${best.recall.toFixed(3)}</td><td>${best.quality.toFixed(3)}</td></tr>`;
    }).join('');
    readout.innerHTML = `<table class="gen-table"><thead><tr><th>index</th>`
      + `<th>distances for 90% recall</th><th>against exhaustive</th>`
      + `<th>best recall reached</th><th>and its category share</th></tr></thead>`
      + `<tbody>${rows}</tbody></table>`
      + `<div class="gen-note">Exhaustive search over ${I.n.toLocaleString('en-US')} vectors `
      + `costs ${Math.round(I.brute_cost)} distances per query and gets `
      + `${I.exact_quality.toFixed(3)} of the ten in the right category, which is the ceiling `
      + `for the second view and is a long way below one: the true nearest neighbours of a `
      + `newswire are not all from its category, and no index can fix that. That is why the `
      + `two views have different shapes. Losing ten per cent of the true neighbours does not `
      + `cost ten per cent of the answer, because the neighbours that are easy to find are `
      + `also the ones that agree with the query.</div>`;
  }

  render();
  return { render, state: st };
}
