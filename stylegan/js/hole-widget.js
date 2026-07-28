/* The hole, and who falls into it.
 *
 * The data has a rectangle of factor space that never occurs: a sprite is
 * never both large and blue. That is the whole argument of this article in one
 * picture, because a generator that maps a gaussian directly onto images has
 * to bend the space to leave that rectangle empty, and bending is measurable.
 * Every point below is a generated sprite measured with the same arithmetic
 * the generator used, so the count in the readout is a count and not an
 * estimate.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { loadSheet, drawTiles } from './panels.js';

const NAME = {
  real: 'the data',
  plain: 'latent at the input',
  styled: 'latent at every layer',
};

export function initHoleWidget(data, sheets) {
  const M = data.meta;
  const state = { which: 'real' };
  const c = makeChart('#hole-chart', {
    width: 520, height: 420, margin: { top: 34, right: 26, bottom: 56, left: 66 },
  });
  const x = d3.scaleLinear().domain([2, M.size_hi + 2]).range([0, c.w]);
  const y = d3.scaleLinear().domain([0, 1]).range([c.h, 0]);
  drawGrid(c.g, x, y, c.w, c.h, { xTicks: 6, yTicks: 5 });
  const holeRect = c.g.append('rect')
    .attr('x', x(M.big)).attr('width', Math.max(0, c.w - x(M.big)))
    .attr('y', y(M.blue[1])).attr('height', y(M.blue[0]) - y(M.blue[1]))
    .attr('fill', 'var(--cosmos)').attr('fill-opacity', 0.12)
    .attr('stroke', 'var(--cosmos)').attr('stroke-dasharray', '4 3');
  const pts = c.g.append('g');
  drawAxes(c.g, x, y, c.w, c.h, { xTicks: 6, yTicks: 5 });
  axisLabels(c.g, c.w, c.h, { x: 'measured size, in pixels', y: 'measured hue' });
  c.g.append('text').attr('class', 'chart-annotation')
    .attr('x', x(M.big) + 8).attr('y', y((M.blue[0] + M.blue[1]) / 2))
    .style('font-size', '11px').style('letter-spacing', '0.4px').style('fill', 'var(--cosmos)')
    .text('large and blue: never in the data');

  function render() {
    const rows = data.scatter[state.which] || [];
    const sel = pts.selectAll('circle').data(rows);
    sel.enter().append('circle').merge(sel)
      .attr('cx', (p) => x(p[0])).attr('cy', (p) => y(p[1])).attr('r', 2)
      .attr('fill', state.which === 'real' ? 'var(--squidink)' : 'var(--primary)')
      .attr('fill-opacity', 0.5);
    sel.exit().remove();

    const inHole = rows.filter((p) => p[0] > M.big && p[1] >= M.blue[0] && p[1] < M.blue[1]).length;
    const share = rows.length ? inHole / rows.length : 0;
    if (state.which === 'real') {
      document.querySelector('#hole-readout').innerHTML =
        `<p>${rows.length} sprites drawn by the generator, measured back off their own pixels. `
        + `${inHole} of them are in the forbidden rectangle, which is what it means for the `
        + `rectangle to be forbidden. Both marginals are flat: `
        + `${(data.dataset.hole.big_share * 100).toFixed(1)}% of sprites are large and `
        + `${(data.dataset.hole.blue_share * 100).toFixed(1)}% are blue, so if the two factors `
        + `were independent, ${(data.dataset.hole.expected_joint * 100).toFixed(1)}% would be `
        + `both. The hole is in the joint distribution only, which is the hard kind to notice `
        + `and the hard kind to reproduce.</p>`;
    } else {
      const ref = data.metrics[`${state.which}/${M.seeds[0]}`].hole;
      document.querySelector('#hole-readout').innerHTML =
        `<p><span class="bold">${NAME[state.which]}</span>: `
        + `<span class="bold">${(ref.share * 100).toFixed(2)}%</span> of what it makes lands in `
        + `the rectangle the data never occupies, against `
        + `${(ref.expected_if_independent * 100).toFixed(2)}% if it had simply forgotten that the `
        + `two factors are related and ${(0).toFixed(2)}% for the data itself. It produces large `
        + `sprites ${(ref.big * 100).toFixed(1)}% of the time and blue ones `
        + `${(ref.blue * 100).toFixed(1)}% of the time.</p>`;
    }
  }

  const row = document.querySelector('#hole-buttons');
  const keys = ['real', 'plain', 'styled'];
  const buttons = keys.map((k) => {
    const b = document.createElement('button');
    b.className = k === state.which ? 'atlas-btn' : 'atlas-btn ghost';
    b.textContent = NAME[k];
    b.addEventListener('click', () => {
      state.which = k;
      buttons.forEach((o, i) => {
        o.className = keys[i] === k ? 'atlas-btn' : 'atlas-btn ghost';
      });
      render();
      drawTiles(document.querySelector('#hole-sheet'), sheets[k].tiles.slice(0, 16), M.tile, 16,
        { zoom: 2, gap: 3 });
    });
    row.appendChild(b);
    return b;
  });
  drawTiles(document.querySelector('#hole-sheet'), sheets.real.tiles.slice(0, 16), M.tile, 16,
    { zoom: 2, gap: 3 });
  render();
  return { state, render };
}

export async function loadSprites(meta) {
  const out = {};
  await Promise.all(['real', 'plain', 'styled'].map(async (k) => {
    out[k] = await loadSheet(`./img/${k}.png`, meta.tile, meta.sheet_cols);
  }));
  return out;
}
