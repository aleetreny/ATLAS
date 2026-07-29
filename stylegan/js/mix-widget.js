/* Style mixing, live: the article's centrepiece.
 *
 * The generator here is the trained one, running in this tab from its float16
 * weights, and the only thing the controls do is decide which style vector
 * each of its five blocks receives. Take two samples, hand block by block the
 * second one's style, and watch which part of the picture moves. Everything
 * under the pictures is measured off the pixels with the same arithmetic the
 * generator used offline, so what you read here is the same quantity as the
 * table above and not a description of it.
 */
import { makeChart, drawAxes, axisLabels } from '../../assets/js/chart.js';
import { drawTiles } from './panels.js';
import { measure, hueDist, makeNoise } from './stylekit.js';

const KEY_NAME = { cx: 'position across', cy: 'position down', size: 'size', hue: 'colour' };

function randn(rng) {
  const u1 = Math.max(rng(), 1e-12);
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

export function initMixWidget(data, gen) {
  const M = data.meta;
  const nb = M.blocks.length;
  const state = { from: nb, zA: null, zB: null, seed: 1 };

  function newZ(mult) {
    let s = (Date.now() % 100000) * mult + 12345;
    const rng = () => {
      s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
      return s / 4294967296;
    };
    const z = new Float32Array(M.z);
    for (let i = 0; i < M.z; i++) z[i] = randn(rng);
    return z;
  }

  function render() {
    const wA = gen.map(state.zA);
    const wB = gen.map(state.zB);
    const ws = [];
    for (let i = 0; i < nb; i++) ws.push(i >= state.from ? wB : wA);
    /* one fixed noise for all three, so the only thing that differs between
       the pictures is the style: two changes at once would make the table
       under them measure the wrong comparison */
    const noise = makeNoise(M.blocks, 20250728);
    const imgA = gen.synth(new Array(nb).fill(wA), noise);
    const imgB = gen.synth(new Array(nb).fill(wB), noise);
    const imgM = gen.synth(ws, noise);
    const caps = ['A', state.from >= nb ? 'A again' : `A, blocks ${state.from + 1} to ${nb} from B`, 'B'];
    drawTiles(document.querySelector('#mix-tiles'), [imgA, imgM, imgB], M.res, 3,
      { zoom: 4, gap: 10, caps });

    const mA = measure(imgA, M.res);
    const mB = measure(imgB, M.res);
    const mM = measure(imgM, M.res);
    /* Same rule as the four hundred pair table: if these two samples barely
       differ in a factor, "how far did it travel" divides by nothing and
       returns noise, so the cell says so instead of printing a percentage. */
    const VIS = data.mixing.visible || { cx: 1, cy: 1, size: 1, hue: 0.1 };
    const rows = ['cx', 'cy', 'size', 'hue'].map((k) => {
      const span = k === 'hue' ? hueDist(mA[k], mB[k]) : Math.abs(mA[k] - mB[k]);
      const moved = k === 'hue' ? hueDist(mA[k], mM[k]) : Math.abs(mA[k] - mM[k]);
      const askable = span > VIS[k];
      const fmt = (v) => (k === 'hue' ? v.toFixed(3) : v.toFixed(2));
      return `<tr><td>${KEY_NAME[k]}</td><td>${fmt(mA[k])}</td><td>${fmt(mM[k])}</td>`
        + `<td>${fmt(mB[k])}</td>`
        + `<td>${askable ? `${((moved / span) * 100).toFixed(0)}%`
          : '<span style="color:var(--stone)">A and B agree on it</span>'}</td></tr>`;
    }).join('');
    document.querySelector('#mix-readout').innerHTML =
      `<p>Blocks ${state.from >= nb ? 'none' : `${state.from + 1} to ${nb}`} take their style from `
      + `B. Everything below is measured off the three pictures above, in this tab.</p>`
      + `<table class="atlas-table"><thead><tr><th>Factor</th><th>in A</th><th>in the mix</th>`
      + `<th>in B</th><th>how far the mix moved towards B</th></tr></thead>`
      + `<tbody>${rows}</tbody></table>`;
  }

  const slider = document.querySelector('#mix-from');
  slider.min = 0;
  slider.max = nb;
  slider.step = 1;
  slider.value = nb;
  const sync = () => {
    document.querySelector('#mix-from-value').textContent =
      state.from >= nb ? 'none' : `${state.from + 1}-${nb}`;
  };
  slider.addEventListener('input', () => {
    state.from = +slider.value;
    sync();
    render();
  });

  const row = document.querySelector('#mix-buttons');
  const mk = (label, fn) => {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = label;
    b.addEventListener('click', fn);
    row.appendChild(b);
  };
  mk('new A', () => { state.zA = newZ(3); render(); });
  mk('new B', () => { state.zB = newZ(7); render(); });

  /* The pair the widget opens on is chosen rather than taken: two samples that
     happen to agree on size and position have nothing to show when a block's
     style is swapped, and the first thing a reader sees would be three rows
     saying so. The search is over a fixed list from a fixed seed, so every
     reader opens on the same two sprites, and it only picks WHICH pair is on
     screen, never what the arithmetic under it says. */
  function pickPair() {
    const VIS = data.mixing.visible || { cx: 1, cy: 1, size: 1, hue: 0.1 };
    const noise = makeNoise(M.blocks, 20250728);
    let s = 20250728;
    const rng = () => {
      s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
      return s / 4294967296;
    };
    const cand = [];
    for (let c = 0; c < 12; c++) {
      const z = new Float32Array(M.z);
      for (let i = 0; i < M.z; i++) z[i] = randn(rng);
      const w = gen.map(z);
      cand.push({ z, m: measure(gen.synth(new Array(nb).fill(w), noise), M.res) });
    }
    let best = { score: -1, i: 0, j: 1 };
    for (let i = 0; i < cand.length; i++) {
      for (let j = i + 1; j < cand.length; j++) {
        let n = 0;
        let sum = 0;
        ['cx', 'cy', 'size', 'hue'].forEach((k) => {
          const d = k === 'hue' ? hueDist(cand[i].m[k], cand[j].m[k])
            : Math.abs(cand[i].m[k] - cand[j].m[k]);
          if (d > VIS[k]) n++;
          sum += Math.min(d / VIS[k], 4);
        });
        const score = n * 10 + sum;
        if (score > best.score) best = { score, i, j };
      }
    }
    return [cand[best.i].z, cand[best.j].z];
  }

  [state.zA, state.zB] = pickPair();
  state.from = 3;
  slider.value = 3;
  sync();
  render();
  return { state, render };
}

/* The measured attribution table: one row per block, one column per factor,
 * from four hundred pairs rather than from the one on screen. */
export function initMixTable(data) {
  const rows = data.mixing.rows;
  const keys = ['cx', 'cy', 'size', 'hue', 'shape'];
  const colours = ['var(--anchor)', 'var(--aqua)', 'var(--primary)', 'var(--cosmos)', 'var(--galaxy)'];
  const label = { cx: 'position across', cy: 'position down', size: 'size', hue: 'colour', shape: 'shape' };
  /* A factor can travel PAST the other sample's value, and that overshoot is
     the interesting part of the measurement rather than an error to hide, so
     the axis is sized to whatever came out and the "all the way" line is drawn
     on it instead of being the top of the chart. */
  const top = Math.max(1, ...rows.flatMap((r) => keys.map((k) => r[k] || 0)));
  const c = makeChart('#mix-chart', {
    width: 620, height: 340, margin: { top: 40, right: 130, bottom: 56, left: 66 },
  });
  const x = d3.scaleBand().domain(rows.map((r) => r.layer)).range([0, c.w]).padding(0.25);
  const y = d3.scaleLinear().domain([0, top * 1.08]).range([c.h, 0]);
  /* the block number goes in the tick, because two of the five blocks run at
     the same resolution and "8px" twice on an axis reads as a mistake */
  drawAxes(c.g, x, y, c.w, c.h, {
    yTicks: 5, yFmt: d3.format('.0%'), xFmt: (d) => `${rows[d].layer + 1}: ${rows[d].res}px`,
  });
  axisLabels(c.g, c.w, c.h, {
    x: 'which block took the other sample\'s style',
    y: 'how far that factor moved',
  });
  c.g.append('line').attr('x1', 0).attr('x2', c.w).attr('y1', y(1)).attr('y2', y(1))
    .attr('stroke', 'var(--stone)').attr('stroke-width', 1.2).attr('stroke-dasharray', '4 4');
  c.g.append('text').attr('class', 'chart-annotation').attr('x', 4).attr('y', y(1) - 6)
    .style('font-size', '11px').style('letter-spacing', '0.4px').style('fill', 'var(--stone)')
    .text('all the way to the other sample');
  const ends = [];
  keys.forEach((k, i) => {
    const pts = rows.filter((r) => r[k] !== null && r[k] !== undefined);
    if (!pts.length) return;
    c.g.append('path').attr('fill', 'none').attr('stroke', colours[i]).attr('stroke-width', 2.2)
      .attr('d', d3.line().x((r) => x(r.layer) + x.bandwidth() / 2).y((r) => y(r[k]))(pts));
    c.g.selectAll(`circle.k${i}`).data(pts).enter().append('circle').attr('class', `k${i}`)
      .attr('cx', (r) => x(r.layer) + x.bandwidth() / 2).attr('cy', (r) => y(r[k])).attr('r', 3)
      .attr('fill', colours[i]);
    ends.push({ k, i, y: y(pts[pts.length - 1][k]) });
  });
  ends.sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) {
    if (ends[i].y - ends[i - 1].y < 14) ends[i].y = ends[i - 1].y + 14;
  }
  ends.forEach((e) => {
    c.g.append('text').attr('class', 'chart-annotation').attr('x', c.w + 6).attr('y', e.y + 4)
      .style('font-size', '11.5px').style('letter-spacing', '0.4px').style('fill', colours[e.i])
      .text(label[e.k]);
  });

  const used = data.mixing.pairs_used || {};
  const strongest = keys.map((k) => {
    const pts = rows.filter((r) => r[k] !== null && r[k] !== undefined);
    return pts.length ? { k, best: pts.slice().sort((a, b) => b[k] - a[k])[0], pts } : null;
  }).filter(Boolean);
  document.querySelector('#mix-chart-readout').innerHTML =
    `<p>One block at a time, over ${data.mixing.n} pairs of samples. A point at 100% means that `
    + `swapping that block's style moved the factor all the way to the other sample's value; a `
    + `point at 0% means the factor did not notice; above 100% it went past. Each factor is scored `
    + `only on the pairs whose two samples differ in it by something visible, a pixel of geometry `
    + `or a tenth of the hue circle, because "how far did the size travel" has no answer for two `
    + `sprites of the same size. The shape line is different in kind: it is the `
    + `fraction of the pairs where the mix came out as the <span class="bold">other</span> `
    + `sample's shape, counted by the classifier, over the pairs whose two shapes differ.</p>`
    + `<table class="atlas-table"><thead><tr><th>Factor</th><th>Block that moves it most</th>`
    + `<th>How far</th><th>Least</th><th>Pairs it could be asked of</th></tr></thead><tbody>`
    + strongest.map((s) => `<tr><td>${label[s.k]}</td>`
      + `<td>block ${s.best.layer + 1}, ${s.best.res} by ${s.best.res}</td>`
      + `<td>${(s.best[s.k] * 100).toFixed(0)}%</td>`
      + `<td>${(Math.min(...s.pts.map((r) => r[s.k])) * 100).toFixed(0)}%</td>`
      + `<td>${used[s.k] === undefined ? '' : `${(used[s.k] * 100).toFixed(0)}%`}</td></tr>`).join('')
    + `</tbody></table>`;
}
