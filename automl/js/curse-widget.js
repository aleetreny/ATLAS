/* How much of the winning score is the winning, on four tables.
 *
 * One curve per table: try b pipelines at random from the catalogue, keep the
 * one with the best cross validated score, and plot how far that score sits
 * above what the same pipeline gets on held out rows. The curve rises with b
 * because the maximum of more noisy numbers is further above the truth, and
 * that is the entire reason a leaderboard with many entries needs a held out
 * set nobody has touched.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

const COLOURS = ['var(--primary)', 'var(--anchor)', 'var(--cosmos)', 'var(--aqua)'];

export function initCurseWidget(data) {
  const T = data.tables;
  const names = Object.keys(T);
  const buttons = document.querySelector('#curse-buttons');
  const readout = document.querySelector('#curse-readout');
  const st = { view: 'gap' };

  const chart = makeChart('#curse-chart', {
    width: 640, height: 380, margin: { top: 56, right: 30, bottom: 58, left: 82 },
  });
  const { g, w, h } = chart;
  const layer = g.append('g');

  const btns = {};
  for (const [key, text] of [['gap', 'how much the score overstates'],
    ['test', 'what the pick is actually worth']]) {
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
    const gap = st.view === 'gap';

    const bs = T[names[0]].curse.map((r) => r.b);
    const x = d3.scaleLog().domain([bs[0], bs[bs.length - 1]]).range([0, w]);
    const all = names.flatMap((n) => T[n].curse.map((r) => (gap ? r.gap : r.test)));
    const y = d3.scaleLinear()
      .domain(gap ? [Math.min(0, d3.min(all)) - 0.004, d3.max(all) + 0.004]
        : [d3.min(all) - 0.02, 1.005])
      .range([h, 0]);
    drawGrid(g, x, y, w, h, { xValues: bs, yTicks: 5 });
    drawAxes(g, x, y, w, h, {
      xValues: bs, yTicks: 5, xFmt: (v) => String(v),
      yFmt: gap ? d3.format('+.3f') : d3.format('.2f'),
    });
    axisLabels(g, w, h, {
      x: 'pipelines the search was allowed to try',
      y: gap ? 'cross validated minus held out' : 'held out accuracy of the pick',
    });

    if (gap) {
      layer.append('line').attr('x1', 0).attr('x2', w).attr('y1', y(0)).attr('y2', y(0))
        .attr('stroke', 'var(--squidink)').attr('stroke-width', 1).attr('opacity', 0.45);
    }
    names.forEach((n, i) => {
      const rows = T[n].curse;
      layer.append('path')
        .attr('d', d3.line().x((r) => x(r.b)).y((r) => y(gap ? r.gap : r.test))(rows))
        .attr('fill', 'none').attr('stroke', COLOURS[i]).attr('stroke-width', 2.2);
      layer.append('text').attr('class', 'chart-note').attr('x', 4 + i * (w / names.length))
        .attr('y', -40).style('font-size', '11px').attr('fill', COLOURS[i]).text(n);
    });

    const rows = names.map((n) => {
      const c = T[n].curse;
      const one = c[0];
      const many = c[c.length - 1];
      return `<tr><td>${n}</td><td>${one.gap >= 0 ? '+' : ''}${one.gap.toFixed(4)}</td>`
        + `<td>${many.gap >= 0 ? '+' : ''}${many.gap.toFixed(4)}</td>`
        + `<td>${(many.gap - one.gap >= 0 ? '+' : '')}${(many.gap - one.gap).toFixed(4)}</td>`
        + `<td>${many.test.toFixed(4)}</td></tr>`;
    }).join('');
    const N = data.meta.configs;
    readout.innerHTML = `<table class="gen-table"><thead><tr><th>table</th>`
      + `<th>gap with 1 try</th><th>gap with ${N}</th><th>what looking cost</th>`
      + `<th>held out at ${N}</th></tr></thead><tbody>${rows}</tbody></table>`
      + `<div class="gen-note">The third column is the part of the reported score that `
      + `exists only because the catalogue was searched. It is not the model getting `
      + `worse: the same pipeline, scored on rows nobody chose it with, is what the last `
      + `column says.</div>`;
  }

  render();
  return { render, state: st };
}
