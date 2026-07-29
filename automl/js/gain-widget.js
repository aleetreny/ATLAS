/* What the search found, against what it started from, on four tables.
 *
 * Three marks per table and they answer three different questions. The bar is
 * the pipeline that comes out of the box. The ring is the one the search picked
 * by cross validation, scored on data it never saw. The faint tick is the best
 * held out score in the whole catalogue, which nobody can reach because nobody
 * is allowed to look. The gap between the ring and the tick is not a failure of
 * the search: it is the price of having to choose blind.
 *
 * The band behind each row is the bootstrap spread of that table's held out
 * set. Differences inside it are differences this page cannot resolve, and two
 * of the four are.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initGainWidget(data) {
  const T = data.tables;
  const names = Object.keys(T);
  const buttons = document.querySelector('#gain-buttons');
  const readout = document.querySelector('#gain-readout');
  const st = { view: 'test' };

  const chart = makeChart('#gain-chart', {
    width: 640, height: 360, margin: { top: 56, right: 34, bottom: 56, left: 92 },
  });
  const { g, w, h } = chart;
  const layer = g.append('g');

  const btns = {};
  for (const [key, text] of [['test', 'scored on held out rows'],
    ['cv', 'scored by cross validation']]) {
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
    const test = st.view === 'test';

    const y = d3.scaleBand().domain(names).range([0, h]).padding(0.42);
    const vals = names.flatMap((n) => [T[n].default[st.view], T[n].best[st.view],
      T[n].span.test_lo]);
    const lo = Math.max(0, d3.min(vals) - 0.04);
    const x = d3.scaleLinear().domain([lo, 1.0]).range([0, w]);
    drawGrid(g, x, y, w, h, { yValues: [], xTicks: 5 });
    drawAxes(g, x, y, w, h, { xTicks: 5, xFmt: d3.format('.2f') });
    axisLabels(g, w, h, { x: 'accuracy', y: '' });

    names.forEach((n) => {
      const row = T[n];
      const yy = y(n);
      const bh = y.bandwidth();
      if (test) {
        const sd = row.best.test_sd;
        layer.append('rect').attr('y', yy - 2).attr('height', bh + 4)
          .attr('x', x(row.best.test - sd))
          .attr('width', Math.max(1, x(row.best.test + sd) - x(row.best.test - sd)))
          .attr('fill', 'var(--anchor)').attr('opacity', 0.13);
      }
      layer.append('rect').attr('x', x(lo)).attr('y', yy)
        .attr('width', Math.max(0, x(row.default[st.view]) - x(lo)))
        .attr('height', bh).attr('fill', 'var(--squidink)').attr('opacity', 0.35);
      layer.append('circle').attr('cx', x(row.best[st.view])).attr('cy', yy + bh / 2)
        .attr('r', 6).attr('fill', 'none')
        .attr('stroke', 'var(--primary)').attr('stroke-width', 2.6);
      if (test) {
        layer.append('line').attr('x1', x(row.best_test.test)).attr('x2', x(row.best_test.test))
          .attr('y1', yy - 3).attr('y2', yy + bh + 3)
          .attr('stroke', 'var(--cosmos)').attr('stroke-width', 2);
      }
    });

    const legend = test
      ? [['out of the box', 'var(--squidink)'], ['what the search picked', 'var(--primary)'],
        ['the best in the catalogue, unreachable', 'var(--cosmos)']]
      : [['out of the box', 'var(--squidink)'], ['what the search picked', 'var(--primary)']];
    legend.forEach(([t, colour], i) => {
      layer.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -42 + i * 15)
        .style('font-size', '11px').attr('fill', colour).text(t);
    });

    const rows = names.map((n) => {
      const d = T[n];
      const gain = d.best.test - d.default.test;
      const real = Math.abs(gain) > d.best.test_sd + d.default.test_sd;
      return `<tr><td>${n}</td><td>${d.default.test.toFixed(4)}</td>`
        + `<td>${d.best.test.toFixed(4)}</td>`
        + `<td>${gain >= 0 ? '+' : ''}${gain.toFixed(4)}</td>`
        + `<td>${real ? 'outside the noise' : 'inside the noise'}</td></tr>`;
    }).join('');
    readout.innerHTML = `<table class="gen-table"><thead><tr><th>table</th>`
      + `<th>out of the box</th><th>searched</th><th>bought</th><th>verdict</th></tr></thead>`
      + `<tbody>${rows}</tbody></table>`
      + `<div class="gen-note">The last column compares the gain against the two `
      + `bootstrap spreads it is made of, which is the only honest bar to clear: a `
      + `difference smaller than how much the held out set itself wobbles is not a `
      + `difference.</div>`;
  }

  render();
  return { render, state: st };
}
