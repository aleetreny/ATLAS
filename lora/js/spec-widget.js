/* The singular spectrum of the update, with the control that gives it meaning.
 *
 * Three curves per layer: the change fine tuning made, the pretrained matrix it
 * was made to, and a random matrix of the same shape scaled to the same
 * Frobenius norm. The third one is the whole argument. "Ninety per cent of the
 * energy in r directions" is a statement about a number of directions and a
 * matrix size, and a random matrix already concentrates some of its energy at
 * the top simply because singular values are sorted. Without the null curve on
 * the same axes there is no way to know whether the update is low rank or
 * whether every matrix of that shape looks like that.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

const SERIES = [
  ['delta', 'what fine tuning changed', 'var(--primary)', null],
  ['base', 'the pretrained matrix', 'var(--anchor)', '6 3'],
  ['random', 'a random matrix, same norm', 'var(--squidink)', '2 3'],
];

export function initSpecWidget(data) {
  const S = data.spectra;
  const buttons = document.querySelector('#spec-buttons');
  const readout = document.querySelector('#spec-readout');
  const st = { layer: 0, view: 'energy' };

  const chart = makeChart('#spec-chart', {
    width: 640, height: 400, margin: { top: 60, right: 30, bottom: 58, left: 78 },
  });
  const { g, w, h } = chart;
  const layer = g.append('g');

  const btns = {};
  S.layers.forEach((L, i) => {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = `layer ${L.layer + 1}, ${L.shape[0]} by ${L.shape[1]}`;
    b.addEventListener('click', () => { st.layer = i; render(); });
    buttons.appendChild(b);
    btns[`L${i}`] = b;
  });
  for (const [key, text] of [['energy', 'energy captured'], ['sv', 'the singular values']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', () => { st.view = key; render(); });
    buttons.appendChild(b);
    btns[key] = b;
  }

  function render() {
    S.layers.forEach((_, i) => btns[`L${i}`].classList.toggle('ghost', i !== st.layer));
    ['energy', 'sv'].forEach((k) => btns[k].classList.toggle('ghost', k !== st.view));
    layer.selectAll('*').remove();
    const L = S.layers[st.layer];
    const energy = st.view === 'energy';
    const n = L.delta.energy.length;

    const x = d3.scaleLinear().domain([1, n]).range([0, w]);
    let y;
    if (energy) {
      y = d3.scaleLinear().domain([0, 1.02]).range([h, 0]);
    } else {
      const all = SERIES.flatMap(([k]) => L[k].s).filter((v) => v > 0);
      y = d3.scaleLog().domain([d3.min(all) * 0.7, d3.max(all) * 1.4]).range([h, 0]);
    }
    drawGrid(g, x, y, w, h, { xTicks: 6, yTicks: 5 });
    drawAxes(g, x, y, w, h, {
      xTicks: 6, yTicks: 5,
      yFmt: energy ? d3.format('.0%') : d3.format('.1e'),
    });
    axisLabels(g, w, h, {
      x: `the first ${n} directions, largest first`,
      y: energy ? 'share of the total energy' : 'singular value',
    });

    if (energy) {
      layer.append('line').attr('x1', 0).attr('x2', w).attr('y1', y(0.9)).attr('y2', y(0.9))
        .attr('stroke', 'var(--smile)').attr('stroke-width', 1.3).attr('stroke-dasharray', '5 3');
      layer.append('text').attr('class', 'chart-note').attr('x', w - 2).attr('y', y(0.9) + 14)
        .attr('text-anchor', 'end').style('font-size', '11px').attr('fill', 'var(--smile)')
        .text('ninety per cent');
    }

    SERIES.forEach(([key, , colour, dash]) => {
      const vals = energy ? L[key].energy : L[key].s;
      layer.append('path')
        .attr('d', d3.line().x((_, i) => x(i + 1)).y((v) => y(Math.max(v, 1e-12)))(vals))
        .attr('fill', 'none').attr('stroke', colour).attr('stroke-width', 2.1)
        .attr('stroke-dasharray', dash);
    });
    SERIES.forEach(([, name, colour, dash], i) => {
      const ly = -44 + i * 15;
      layer.append('line').attr('x1', 0).attr('x2', 22).attr('y1', ly - 4).attr('y2', ly - 4)
        .attr('stroke', colour).attr('stroke-width', 2.4).attr('stroke-dasharray', dash);
      layer.append('text').attr('class', 'chart-note').attr('x', 28).attr('y', ly)
        .style('font-size', '11px').attr('fill', colour).text(name);
    });

    const lowRank = L.delta.r90 < L.random.r90;
    readout.innerHTML = `<table class="gen-table"><thead><tr><th></th>`
      + `<th>directions for 90%</th><th>for 50%</th><th>effective rank</th></tr></thead>`
      + `<tbody>`
      + `<tr><td>the update</td><td>${L.delta.r90}</td><td>${L.delta.r50}</td>`
      + `<td>${L.delta.eff.toFixed(1)}</td></tr>`
      + `<tr><td>the pretrained matrix</td><td>${L.base.r90}</td><td></td>`
      + `<td>${L.base.eff.toFixed(1)}</td></tr>`
      + `<tr><td>a random matrix</td><td>${L.random.r90}</td><td></td>`
      + `<td>${L.random.eff.toFixed(1)}</td></tr>`
      + `</tbody></table><div class="gen-note">On this layer the update needs `
      + `<span class="bold">${L.delta.r90}</span> directions to hold ninety per cent of its `
      + `energy; a random matrix of the same shape and the same norm needs `
      + `${L.random.r90}, and the pretrained matrix itself needs ${L.base.r90}. `
      + (Math.max(L.base.r90, L.random.r90) > n
        ? `Those last two are off the right of the plot on purpose: the chart draws the first `
          + `${n} directions, and neither of them has reached the dashed line by then. `
        : '')
      + (lowRank
        ? `So the update really is concentrated: ${(L.random.r90 / L.delta.r90).toFixed(1)} `
          + `times fewer directions than the null. `
        : `So on this layer the update is <span class="bold">not</span> more concentrated `
          + `than a random matrix, and an adapter of small rank has no reason to work here. `)
      + `And the change itself is not small on this model: its norm is `
      + `${L.delta.fro.toFixed(2)}, which is `
      + `${(L.rel * 100).toFixed(1)}% of the norm of the matrix it was added to, where the `
      + `usual description of fine tuning as a nudge would put it in the low single `
      + `digits.</div>`;
  }

  render();
  return { render, state: st };
}
