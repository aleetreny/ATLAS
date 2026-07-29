/* Grid against random at the same budget, on three surfaces.
 *
 * The third surface is the one that makes this an experiment rather than an
 * anecdote. It is built from the same 961 measured numbers with one axis
 * replaced by its own maximum, so that axis moves nothing and the other keeps
 * exactly the shape it was measured to have. If the usual explanation for why
 * random search wins is right, that is where it has to win.
 *
 * The picture shows where each search looks; the readout carries the scores,
 * which are averages over four hundred draws and not over the one draw drawn.
 */
import { makeChart, drawAxes, axisLabels } from '../../assets/js/chart.js';

export function initSearchWidget(data) {
  const C = data.compare;
  const buttons = document.querySelector('#search-buttons');
  const readout = document.querySelector('#search-readout');
  const slider = document.querySelector('#search-slider');
  const label = document.querySelector('#search-value');
  const st = { view: 'svm', b: 3 };

  const SURFACES = {
    svm: { rows: C.svm, size: data.surface.cv.length, name: 'the kernel machine',
      importance: data.surface.importance },
    forest: { rows: C.forest, size: data.forest.cv.length, name: 'the forest',
      importance: data.forest.importance,
      axes: [`depth ${data.forest.depth[0]} to `
        + `${data.forest.depth[data.forest.depth.length - 1]}, ${data.forest.trees} trees`,
      `${data.forest.feats[0]} to `
        + `${data.forest.feats[data.forest.feats.length - 1]} columns per split`] },
    flattened: { rows: C.flattened, size: data.surface.cv.length,
      name: 'one axis switched off', importance: C.flat_importance },
  };

  const chart = makeChart('#search-chart', {
    width: 640, height: 380, margin: { top: 44, right: 30, bottom: 56, left: 66 },
  });
  const { g, w, h } = chart;
  const layer = g.append('g');

  const btns = {};
  for (const [key, text] of [['svm', 'the kernel machine'], ['forest', 'the forest'],
    ['flattened', 'one axis switched off']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', () => { st.view = key; sync(); render(); });
    buttons.appendChild(b);
    btns[key] = b;
  }

  function sync() {
    slider.max = String(SURFACES[st.view].rows.length - 1);
    if (st.b > Number(slider.max)) st.b = Number(slider.max);
    slider.value = String(st.b);
  }
  slider.addEventListener('input', () => { st.b = Number(slider.value); render(); });

  /* the same congruential generator on both sides of the site, so the dots the
     reader sees are a real draw and not a decorative scatter */
  function lcg(seed) {
    let s = seed >>> 0;
    return () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; };
  }

  function render() {
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', k !== st.view));
    const S = SURFACES[st.view];
    const row = S.rows[st.b];
    label.textContent = `${row.budget} evaluations`;
    layer.selectAll('*').remove();

    const n = S.size;
    const x = d3.scaleLinear().domain([-0.5, n - 0.5]).range([0, w]);
    const y = d3.scaleLinear().domain([-0.5, n - 0.5]).range([h, 0]);
    drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 5, xFmt: d3.format('d'), yFmt: d3.format('d') });
    axisLabels(g, w, h, { x: 'first dial, as a grid index', y: 'second dial, as a grid index' });

    const k = row.distinct_grid;
    const idx = [];
    for (let i = 0; i < k; i++) idx.push(Math.round(i * (n - 1) / Math.max(1, k - 1)));
    const uniq = Array.from(new Set(idx));
    uniq.forEach((i) => uniq.forEach((j) => {
      layer.append('rect').attr('x', x(j) - 4).attr('y', y(i) - 4)
        .attr('width', 8).attr('height', 8)
        .attr('fill', 'var(--primary)').attr('opacity', 0.85);
    }));

    const rnd = lcg(19 + row.budget * 7);
    for (let t = 0; t < row.budget; t++) {
      const i = Math.floor(rnd() * n);
      const j = Math.floor(rnd() * n);
      layer.append('circle').attr('cx', x(j)).attr('cy', y(i)).attr('r', 4)
        .attr('fill', 'none').attr('stroke', 'var(--cosmos)').attr('stroke-width', 1.8);
    }

    layer.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -24)
      .style('font-size', '11.5px').attr('fill', 'var(--primary)')
      .text(`squares: the lattice, ${row.distinct_grid} values per axis`);
    layer.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -8)
      .style('font-size', '11.5px').attr('fill', 'var(--cosmos)')
      .text(`rings: one random draw, ${row.distinct_rand} values per axis`);

    const floor = data.floor.test_sd;
    const diff = row.rand_test - row.grid_test;
    const inside = Math.abs(diff) <= floor;
    readout.innerHTML = `<div class="gen-note">On ${S.name}, with `
      + `${row.budget} evaluations each. The lattice lands on a held out score of `
      + `<span class="bold">${row.grid_test.toFixed(4)}</span>; the average over 400 `
      + `random draws is <span class="bold">${row.rand_test.toFixed(4)}</span>, with a `
      + `spread of ${row.rand_test_sd.toFixed(4)} between draws and `
      + `${(row.rand_beats * 100).toFixed(1)}% of them beating the lattice. `
      + (inside
        ? `The ${Math.abs(diff).toFixed(4)} between the two is inside the `
          + `${floor.toFixed(4)} that this held out set moves under resampling, so on `
          + `this surface at this budget the choice does not decide anything.`
        : `The ${Math.abs(diff).toFixed(4)} between them is larger than the `
          + `${floor.toFixed(4)} the held out set moves under resampling, and it favours `
          + `${diff > 0 ? 'the random draw' : 'the lattice'}.`)
      + ` Switching one dial off matters more than switching method: on this surface the `
      + `first dial moves the best reachable score by ${S.importance.axis0.toFixed(3)} and `
      + `the second by ${S.importance.axis1.toFixed(3)}`
      + (S.axes ? ` (${S.axes.join(' and ')})` : '')
      + `. On the score that chose them, which is the one a search actually sees, the `
      + `lattice reads ${row.grid_cv.toFixed(4)} and the average random draw reads `
      + `${row.rand_cv.toFixed(4)}: `
      + (row.rand_cv > row.grid_cv
        ? `the random draw looks better there than it turns out to be, which is the `
          + `winner's curse of the last section showing up in a comparison.`
        : `the two agree about as well there as they do on the held out rows.`)
      + `</div>`;
  }

  sync();
  render();
  return { render, state: st };
}
