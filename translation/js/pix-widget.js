/* The paired half: what an averaging loss is obliged to produce, what an
 * adversarial one produces instead, and what happens when you draw your own.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { makeCanvas, paintRGB } from '../../assets/js/imagery.js';
import { drawTiles } from './panels.js';
import { highFreq } from './unetkit.js';

export const ARM_NAME = {
  l1: 'L1 alone',
  gan: 'adversarial alone',
  l1gan: 'both together',
  'patch-pixel': 'one pixel at a time',
  'patch-full': 'the whole image at once',
};
const ARM_COLOUR = {
  l1: 'var(--anchor)',
  gan: 'var(--cosmos)',
  l1gan: 'var(--primary)',
  'patch-pixel': 'var(--smile)',
  'patch-full': 'var(--galaxy)',
};

/* ------------------------------------------------------------------------ */
export function initStrip(data, sheet) {
  const M = data.meta;
  const cols = M.sheet_cols;
  const order = M.sheet_order;
  const rowsWanted = ['plan', 'render', 'l1', 'gan', 'l1gan'];
  const label = {
    plan: 'the plan, which is the input',
    render: 'the render, which is the answer',
    l1: 'L1 alone',
    gan: 'adversarial alone',
    l1gan: 'both together',
  };
  const host = document.querySelector('#strip-rows');
  host.innerHTML = '';
  rowsWanted.forEach((name) => {
    const idx = order.indexOf(name);
    if (idx < 0) return;
    const div = document.createElement('div');
    host.appendChild(div);
    drawTiles(div, sheet.tiles.slice(idx * cols, idx * cols + cols), M.tile, cols,
      { zoom: 3, gap: 4, label: label[name] });
  });
}

/* ------------------------------------------------------------------------ */
export function initCondWidget(data) {
  const C = data.conditional;
  const R = data.real;
  const S = data.scores;
  const seed = data.meta.seeds[0];
  const bars = [
    { k: 'real render', v: R.hf, colour: 'var(--squidink)' },
    { k: 'median render', v: C[0].median_hf, colour: 'var(--stone)' },
  ].concat(data.meta.arms.map((a) => ({
    k: ARM_NAME[a], v: S[`${a}/${seed}`].hf, colour: ARM_COLOUR[a],
  })));

  const c = makeChart('#cond-chart', {
    width: 680, height: 360, margin: { top: 36, right: 26, bottom: 96, left: 76 },
  });
  const x = d3.scaleBand().domain(bars.map((b) => b.k)).range([0, c.w]).padding(0.28);
  const y = d3.scaleLinear().domain([0, d3.max(bars, (b) => b.v) * 1.18]).range([c.h, 0]);
  drawGrid(c.g, x, y, c.w, c.h, { yTicks: 5 });
  drawAxes(c.g, x, y, c.w, c.h, { yTicks: 5 });
  axisLabels(c.g, c.w, c.h, { y: 'detail left after a blur' });
  c.g.selectAll('rect').data(bars).enter().append('rect')
    .attr('x', (b) => x(b.k)).attr('y', (b) => y(b.v))
    .attr('width', x.bandwidth()).attr('height', (b) => c.h - y(b.v))
    .attr('fill', (b) => b.colour);
  c.g.selectAll('text.v').data(bars).enter().append('text').attr('class', 'chart-annotation v')
    .attr('x', (b) => x(b.k) + x.bandwidth() / 2).attr('y', (b) => y(b.v) - 6)
    .attr('text-anchor', 'middle').style('font-size', '11px').style('letter-spacing', '0.3px')
    .text((b) => b.v.toFixed(4));
  c.g.selectAll('.axis.x text')
    .style('font-size', '9px')
    .attr('transform', 'rotate(-42)')
    .attr('text-anchor', 'end')
    .attr('dx', '-4')
    .attr('dy', '6');

  const rows = data.meta.arms.map((a) => {
    const s = S[`${a}/${seed}`];
    const other = S[`${a}/${data.meta.seeds[data.meta.seeds.length - 1]}`];
    return `<tr><td>${ARM_NAME[a]}</td><td>${s.field} px</td><td>${s.l1.toFixed(4)}</td>`
      + `<td>${(s.hf_share * 100).toFixed(1)}%</td><td>${(s.commit_share * 100).toFixed(1)}%</td>`
      + `<td>${Math.abs(s.l1 - other.l1).toFixed(4)}</td></tr>`;
  }).join('');
  document.querySelector('#cond-readout').innerHTML =
    `<table class="atlas-table"><thead><tr><th>Trained with</th><th>Patch the critic sees</th>`
    + `<th>Distance to the true render</th><th>Detail, against a real render</th>`
    + `<th>Windows committed</th><th>Seed spread on the distance</th></tr></thead>`
    + `<tbody>${rows}</tbody></table>`;
}

/* ------------------------------------------------------------------------ */
const TOOLS = [
  { name: 'sky', key: 'sky' },
  { name: 'ground', key: 'ground' },
  { name: 'road', key: 'road' },
  { name: 'building A', key: 'b0' },
  { name: 'building B', key: 'b1' },
];

export function initDrawWidget(data, nets) {
  const M = data.meta;
  const res = M.res;
  const zoom = 8;
  const P = M.plan_colours;
  const colour = {
    sky: P.sky, ground: P.ground, road: P.road, b0: P.body[0], b1: P.body[1],
  };
  const state = { tool: 'b0', plan: new Float32Array(res * res * 3), painting: false };

  const host = document.querySelector('#draw-canvas');
  const { canvas, ctx } = makeCanvas(host, res * zoom, res * zoom);
  canvas.style.cursor = 'crosshair';
  canvas.style.imageRendering = 'pixelated';
  canvas.style.touchAction = 'none';
  const outs = ['l1', 'l1gan'].map((k) => {
    const box = document.createElement('div');
    box.className = 'out-cell';
    document.querySelector('#draw-outputs').appendChild(box);
    const c = makeCanvas(box, res * zoom, res * zoom);
    const cap = document.createElement('div');
    cap.className = 'cap';
    cap.textContent = k === 'l1' ? 'L1 alone' : 'both together';
    box.appendChild(cap);
    c.canvas.style.imageRendering = 'pixelated';
    return { k, ctx: c.ctx };
  });

  function setPixel(px, py, key) {
    const col = colour[key];
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const x = px + dx;
        const y = py + dy;
        if (x < 0 || y < 0 || x >= res || y >= res) continue;
        const p = (y * res + x) * 3;
        for (let c = 0; c < 3; c++) state.plan[p + c] = col[c];
      }
    }
  }

  function randomPlan() {
    const rnd = (a, b) => a + Math.floor(Math.random() * (b - a));
    const hz = rnd(9, 15);
    const ry = Math.min(hz + rnd(4, 9), res - 2);
    const rh = Math.min(rnd(4, 7), res - ry);
    const bx = rnd(2, 17);
    const bw = rnd(7, 13);
    const bh = rnd(7, 14);
    const bt = rnd(0, 2);
    const put = (y0, y1, x0, x1, col) => {
      for (let y = Math.max(y0, 0); y < Math.min(y1, res); y++) {
        for (let x = Math.max(x0, 0); x < Math.min(x1, res); x++) {
          const p = (y * res + x) * 3;
          for (let c = 0; c < 3; c++) state.plan[p + c] = col[c];
        }
      }
    };
    put(0, hz, 0, res, P.sky);
    put(hz, res, 0, res, P.ground);
    put(ry, ry + rh, 0, res, P.road);
    put(hz - bh, hz, bx, bx + bw, P.body[bt]);
  }

  let pending = null;
  function render() {
    paintRGB(ctx, state.plan, res, res, { zoom });
    if (pending) clearTimeout(pending);
    pending = setTimeout(() => {
      const results = outs.map((o) => {
        const img = nets[o.k].run(state.plan, res);
        paintRGB(o.ctx, img, res, res, { zoom });
        return { k: o.k, hf: highFreq(img, res) };
      });
      const real = data.real.hf;
      const toolName = (TOOLS.find((o) => o.key === state.tool) || {}).name || state.tool;
      document.querySelector('#draw-readout').innerHTML =
        `<p>Painting with <span class="bold">${toolName}</span>. Both networks are running on your `
        + `plan, here, from their quantised weights. On this `
        + `input the detail left after a blur is `
        + results.map((r) => `<span class="bold">${r.hf.toFixed(4)}</span> for `
          + `${r.k === 'l1' ? 'L1 alone' : 'the two together'}`).join(' and ')
        + `, against ${real.toFixed(4)} for a real render. The same quantity, computed the same `
        + `way as every number in the table above.</p>`
        + `<p>Draw something the training set never contained and the comparison gets more `
        + `interesting than it is on a normal plan: neither network has ever seen two buildings, `
        + `or a road above the horizon.</p>`;
    }, 60);
  }

  const at = (e) => {
    const r = canvas.getBoundingClientRect();
    return [Math.floor(((e.clientX - r.left) / r.width) * res),
      Math.floor(((e.clientY - r.top) / r.height) * res)];
  };
  canvas.addEventListener('pointerdown', (e) => {
    state.painting = true;
    canvas.setPointerCapture(e.pointerId);
    const [x, y] = at(e);
    setPixel(x, y, state.tool);
    render();
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!state.painting) return;
    const [x, y] = at(e);
    setPixel(x, y, state.tool);
    render();
  });
  canvas.addEventListener('pointerup', () => { state.painting = false; });

  const row = document.querySelector('#draw-tools');
  const buttons = TOOLS.map((t) => {
    const b = document.createElement('button');
    b.className = t.key === state.tool ? 'atlas-btn' : 'atlas-btn ghost';
    b.textContent = t.name;
    b.addEventListener('click', () => {
      state.tool = t.key;
      buttons.forEach((o, i) => {
        o.className = TOOLS[i].key === t.key ? 'atlas-btn' : 'atlas-btn ghost';
      });
      render();
    });
    row.appendChild(b);
    return b;
  });
  const extra = document.querySelector('#draw-buttons');
  [['another plan', () => { randomPlan(); render(); }]].forEach(([label, fn]) => {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = label;
    b.addEventListener('click', fn);
    extra.appendChild(b);
  });

  randomPlan();
  render();
  return { state, render };
}

/* ------------------------------------------------------------------------ */
export function initPatchWidget(data) {
  const F = data.fields;
  const S = data.scores;
  const seed = data.meta.seeds[0];
  const kinds = Object.keys(F.plain);
  const rows = kinds.map((k) => {
    const arm = data.meta.arms.find((a) => S[`${a}/${seed}`].patch === k);
    const s = arm ? S[`${arm}/${seed}`] : null;
    return `<tr><td>${F.specs[k].map((p) => `${p[0]}x${p[0]} stride ${p[1]}`).join(', ')}</td>`
      + `<td>${F.plain[k]} px</td><td>${F.normalised[k]} px</td>`
      + `<td>${arm ? ARM_NAME[arm] : 'not trained'}</td>`
      + `<td>${s ? s.l1.toFixed(4) : ''}</td>`
      + `<td>${s ? `${(s.hf_share * 100).toFixed(1)}%` : ''}</td></tr>`;
  }).join('');
  document.querySelector('#patch-readout').innerHTML =
    `<table class="atlas-table"><thead><tr><th>Discriminator</th><th>Window it really sees</th>`
    + `<th>The same with a normalisation layer</th><th>Used by</th>`
    + `<th>Distance to the render</th><th>Detail</th></tr></thead><tbody>${rows}</tbody></table>`;
}
