/* What a condition on the query does to a graph index.
 *
 * The second view is the mechanism and it is the reason this problem is hard
 * rather than fiddly. A graph search is a walk, and a walk cannot cross into a
 * component it is not in. Keeping only the rows that pass the filter leaves an
 * induced subgraph, and counting its connected components says exactly how much
 * of the collection the walk can still reach. When that count goes up, no
 * amount of widening the search will help.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initFilterWidget(data) {
  const F = data.filter;
  const buttons = document.querySelector('#filter-buttons');
  const readout = document.querySelector('#filter-readout');
  const st = { view: 'recall' };

  const chart = makeChart('#filter-chart', {
    width: 640, height: 380, margin: { top: 58, right: 44, bottom: 58, left: 84 },
  });
  const { g, w, h } = chart;
  const layer = g.append('g');

  const btns = {};
  for (const [key, text] of [['recall', 'what each strategy finds'],
    ['pieces', 'what the filter does to the graph'],
    ['cost', 'what each strategy costs']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', () => { st.view = key; render(); });
    buttons.appendChild(b);
    btns[key] = b;
  }

  const rows = F.rows.slice().sort((a, b) => a.selectivity - b.selectivity);

  function render() {
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', k !== st.view));
    layer.selectAll('*').remove();

    const x = d3.scaleLog().domain([rows[0].selectivity * 0.7, rows[rows.length - 1].selectivity * 1.3])
      .range([0, w]);
    let y;
    if (st.view === 'recall') {
      y = d3.scaleLinear().domain([0, 1.03]).range([h, 0]);
    } else if (st.view === 'pieces') {
      y = d3.scaleLog().domain([0.8, Math.max(...rows.map((r) => r.pieces)) * 1.4]).range([h, 0]);
    } else {
      const costs = rows.flatMap((r) => [r.post_cost, r.over_cost, r.pre_cost]);
      y = d3.scaleLog().domain([Math.min(...costs) * 0.7, Math.max(...costs) * 1.4]).range([h, 0]);
    }
    drawGrid(g, x, y, w, h, { xValues: rows.map((r) => r.selectivity), yTicks: 5 });
    drawAxes(g, x, y, w, h, {
      xValues: rows.map((r) => r.selectivity), yTicks: 5,
      xFmt: (v) => `${(v * 100).toFixed(v < 0.05 ? 0 : 0)}%`,
      yFmt: st.view === 'recall' ? d3.format('.1f')
        : (v) => (v >= 1000 ? `${(v / 1000).toFixed(1)}k` : String(Math.round(v))),
    });
    axisLabels(g, w, h, {
      x: 'share of the collection the filter keeps',
      y: st.view === 'recall' ? 'recall at ten, inside the filter'
        : st.view === 'pieces' ? 'connected pieces of the kept subgraph'
          : 'distances computed per query',
    });

    const series = st.view === 'recall'
      ? [['post_recall', 'var(--cosmos)', 'search then drop what fails the filter'],
        ['over_recall', 'var(--primary)', 'ask for more, then drop'],
        [null, 'var(--anchor)', 'scan the kept rows: always one']]
      : st.view === 'pieces'
        ? [['pieces', 'var(--primary)', 'pieces the walk cannot cross between'],
          ['biggest', 'var(--anchor)', 'share of kept rows in the largest piece']]
        : [['post_cost', 'var(--cosmos)', 'search then drop'],
          ['over_cost', 'var(--primary)', 'ask for more, then drop'],
          ['pre_cost', 'var(--anchor)', 'scan the kept rows']];

    series.forEach(([key, colour, name], i) => {
      if (key === 'biggest') {
        /* a share, not a count: it goes on the recall style axis and is drawn
           as text rather than sharing a log axis with a count */
        return;
      }
      if (key) {
        layer.append('path')
          .attr('d', d3.line().x((r) => x(r.selectivity)).y((r) => y(r[key]))(rows))
          .attr('fill', 'none').attr('stroke', colour).attr('stroke-width', 2.2);
        rows.forEach((r) => layer.append('circle').attr('cx', x(r.selectivity))
          .attr('cy', y(r[key])).attr('r', 4).attr('fill', colour));
      } else {
        layer.append('line').attr('x1', 0).attr('x2', w).attr('y1', y(1)).attr('y2', y(1))
          .attr('stroke', colour).attr('stroke-width', 1.8).attr('stroke-dasharray', '6 4');
      }
      layer.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -46 + i * 15)
        .style('font-size', '11px').attr('fill', colour).text(name);
    });

    const tight = rows[0];
    const loose = rows[rows.length - 1];
    const cats = F.categories.slice().sort((a, b) => a.selectivity - b.selectivity);
    const catRows = cats.map((c) => `<tr><td>${c.category}</td>`
      + `<td>${(c.selectivity * 100).toFixed(1)}%</td><td>${c.post_recall.toFixed(3)}</td>`
      + `<td>${c.pieces}</td><td>${(c.biggest * 100).toFixed(1)}%</td></tr>`).join('');
    readout.innerHTML = `<table class="gen-table"><thead><tr><th>filter</th>`
      + `<th>keeps</th><th>recall, searching then dropping</th><th>pieces</th>`
      + `<th>largest piece</th></tr></thead><tbody>${catRows}</tbody></table>`
      + `<div class="gen-note">At ${(loose.selectivity * 100).toFixed(0)}% selectivity, `
      + `searching the whole index and dropping what fails the filter recalls `
      + `${loose.post_recall.toFixed(3)} of the true neighbours inside the filter and returns `
      + `nothing at all for ${(loose.post_empty * 100).toFixed(1)}% of queries; at `
      + `${(tight.selectivity * 100).toFixed(0)}% it recalls `
      + `${tight.post_recall.toFixed(3)} and returns nothing for `
      + `${(tight.post_empty * 100).toFixed(1)}%. Asking for enough extra to compensate `
      + `brings it to ${tight.over_recall.toFixed(3)} for `
      + `${Math.round(tight.over_cost)} distances against ${Math.round(tight.pre_cost)} for `
      + `simply scanning the kept rows, which is exact. That crossover is the whole design `
      + `decision, and it moves with the selectivity: below some share of the collection, the `
      + `index is worse than not having one. The table above uses a filter that is not `
      + `random, the newswire's own category, which is the realistic case because the rows `
      + `that pass a real filter are not scattered.</div>`;
  }

  render();
  return { render, state: st };
}
