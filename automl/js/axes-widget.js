/* The four decisions in the same units.
 *
 * Each bar is the best score still reachable once that level is nailed down, so
 * the height of a group is what fixing that decision costs at worst. Putting
 * all four on one axis is the whole point: "the preprocessing matters more than
 * the model" is only sayable if both were measured with the same ruler, and
 * almost nobody does that because the two live in different libraries.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

const AXES = ['scaler', 'reducer', 'weight', 'model'];
const TITLE = {
  scaler: 'how the columns are scaled',
  reducer: 'whether the columns are reduced',
  weight: 'whether the classes are weighted',
  model: 'which model, with which setting',
};

export function initAxesWidget(data) {
  const T = data.tables;
  const names = Object.keys(T);
  const buttons = document.querySelector('#axes-buttons');
  const readout = document.querySelector('#axes-readout');
  const st = { table: names[0] };

  const chart = makeChart('#axes-chart', {
    width: 640, height: 420, margin: { top: 40, right: 26, bottom: 92, left: 76 },
  });
  const { g, w, h } = chart;
  const layer = g.append('g');

  const btns = {};
  names.forEach((n) => {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = n;
    b.addEventListener('click', () => { st.table = n; render(); });
    buttons.appendChild(b);
    btns[n] = b;
  });

  function render() {
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', k !== st.table));
    layer.selectAll('*').remove();
    const A = T[st.table].axes;

    const levels = [];
    AXES.forEach((ax) => Object.keys(A[ax].best).forEach((lv) => levels.push([ax, lv])));
    const x = d3.scaleBand().domain(levels.map(([a, l]) => `${a}|${l}`))
      .range([0, w]).padding(0.25);
    const vals = levels.map(([a, l]) => A[a].best[l]);
    const y = d3.scaleLinear().domain([d3.min(vals) - 0.03, d3.max(vals) + 0.01]).range([h, 0]);
    drawGrid(g, x, y, w, h, { xValues: [], yTicks: 5 });
    drawAxes(g, x, y, w, h, { xValues: [], yTicks: 5, yFmt: d3.format('.3f') });
    axisLabels(g, w, h, { x: '', y: 'best score still reachable' });

    const colour = d3.scaleOrdinal().domain(AXES)
      .range(['var(--primary)', 'var(--anchor)', 'var(--aqua)', 'var(--cosmos)']);
    levels.forEach(([ax, lv]) => {
      const key = `${ax}|${lv}`;
      const v = A[ax].best[lv];
      layer.append('rect').attr('x', x(key)).attr('width', x.bandwidth())
        .attr('y', y(v)).attr('height', h - y(v)).attr('fill', colour(ax));
      /* the level name goes under its bar, rotated, because these are words and
         there are up to fifteen of them across a 640 unit canvas */
      layer.append('text').attr('class', 'chart-note')
         .attr('transform', `translate(${x(key) + x.bandwidth() / 2},${h + 8}) rotate(-85)`)
        .attr('text-anchor', 'end').style('font-size', '8px').text(lv.length > 13 ? `${lv.slice(0, 12)}.` : lv);
    });

    AXES.forEach((ax, i) => {
      layer.append('text').attr('class', 'chart-note')
        .attr('x', 4 + i * (w / AXES.length)).attr('y', -22)
        .style('font-size', '11px').attr('fill', colour(ax))
        .text(`${ax}: ${A[ax].span.toFixed(3)}`);
    });

    const ranked = AXES.map((a) => [a, A[a].span]).sort((p, q) => q[1] - p[1]);
    const [top, topSpan] = ranked[0];
    const [bot, botSpan] = ranked[ranked.length - 1];
    readout.innerHTML = `<div class="gen-note">On <span class="bold">${st.table}</span>, `
      + `nailing down <span class="bold">${TITLE[top]}</span> moves the best still `
      + `reachable score by ${topSpan.toFixed(3)}, which is the largest of the four, and `
      + `${TITLE[bot]} moves it by ${botSpan.toFixed(3)}, the smallest. The worst level of `
      + `the first is ${A[top].worst_level} and the best is ${A[top].best_level}. `
      + `The order across the four decisions is `
      + `${ranked.map(([a, s]) => `${a} ${s.toFixed(3)}`).join(', ')}. `
      + `Every one of these bars is the maximum over everything else, so a short bar means `
      + `"the rest of the pipeline can recover from this choice", not "this choice does `
      + `nothing".</div>`;
  }

  render();
  return { render, state: st };
}
