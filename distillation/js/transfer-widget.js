/* The transfer curve with the controls that decide what it means.
 *
 * Six lines, and two of them are the article. "Random" is the same architecture
 * with the weights it was born with: whatever it reaches is what a projection
 * buys, not what learning buys. "Shuffled" is the same architecture trained to
 * convergence on labels that were dealt at random: it has had exactly as much
 * optimiser applied to it as the transferred one and has learned nothing worth
 * carrying, which separates "trained" from "trained on something related".
 *
 * Without those two, the gap between transfer and raw pixels is a number with
 * three explanations and no way to choose between them.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

const ARMS = [
  ['transfer', 'frozen encoder from the other task', 'var(--primary)', null],
  ['oracle', 'frozen encoder from this task', 'var(--smile)', '1 3'],
  ['random', 'same shape, never trained', 'var(--aqua)', '5 3'],
  ['shuffled', 'same shape, trained on shuffled labels', 'var(--cosmos)', '2 3'],
  ['pixels', 'the raw pixels', 'var(--squidink)', '7 4'],
  ['finetune', 'the whole thing fine tuned', 'var(--anchor)', null],
  ['scratch', 'from scratch', 'var(--galaxy)', '6 3'],
];

export function initTransferWidget(data) {
  const buttons = document.querySelector('#transfer-buttons');
  const readout = document.querySelector('#transfer-readout');
  const st = { dir: 'forward' };

  const chart = makeChart('#transfer-chart', {
    width: 640, height: 430, margin: { top: 106, right: 30, bottom: 58, left: 74 },
  });
  const { g, w, h } = chart;
  const layer = g.append('g');

  const btns = {};
  for (const [key, text] of [['forward', 'digits, then clothes'],
    ['backward', 'clothes, then digits']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', () => { st.dir = key; render(); });
    buttons.appendChild(b);
    btns[key] = b;
  }

  function render() {
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', k !== st.dir));
    layer.selectAll('*').remove();
    const D = data.transfer[st.dir];
    const rows = D.rows;

    const x = d3.scaleLog().domain([rows[0].n, rows[rows.length - 1].n]).range([0, w]);
    const all = rows.flatMap((r) => ARMS.map(([k]) => r[k])).filter((v) => v != null);
    const y = d3.scaleLinear().domain([d3.min(all) - 0.03, d3.max(all) + 0.02]).range([h, 0]);
    drawGrid(g, x, y, w, h, { xValues: rows.map((r) => r.n), yTicks: 5 });
    drawAxes(g, x, y, w, h, {
      xValues: rows.map((r) => r.n), yTicks: 5,
      xFmt: (v) => (v >= 1000 ? `${v / 1000}k` : String(v)), yFmt: d3.format('.2f'),
    });
    axisLabels(g, w, h, { x: 'labelled rows of the new task', y: 'accuracy on its test split' });

    ARMS.forEach(([key, , colour, dash]) => {
      layer.append('path')
        .attr('d', d3.line().x((r) => x(r.n)).y((r) => y(r[key]))(rows))
        .attr('fill', 'none').attr('stroke', colour).attr('stroke-width', 2)
        .attr('stroke-dasharray', dash);
    });

    /* the legend is above the plot: seven lines converge on the right hand side,
       which is the one region where seven labels certainly do not fit */
    ARMS.forEach(([, name, colour, dash], i) => {
      const col = i % 2;
      const ly = -92 + Math.floor(i / 2) * 15;
      const lx = col * (w / 2);
      layer.append('line').attr('x1', lx).attr('x2', lx + 20)
        .attr('y1', ly - 4).attr('y2', ly - 4)
        .attr('stroke', colour).attr('stroke-width', 2.4).attr('stroke-dasharray', dash);
      layer.append('text').attr('class', 'chart-note').attr('x', lx + 26).attr('y', ly)
        .style('font-size', '11px').attr('fill', colour).text(name);
    });

    const small = rows[0];
    const big = rows[rows.length - 1];
    const closes = rows.find((r) => r.transfer - r.scratch <= r.transfer_sd + r.scratch_sd);
    readout.innerHTML = `<table class="gen-table"><thead><tr><th>rows</th>`
      + `<th>pixels</th><th>never trained</th><th>shuffled labels</th><th>transferred</th>`
      + `<th>this task</th></tr></thead><tbody>`
      + [small, rows[Math.floor(rows.length / 2)], big].map((r) => `<tr><td>${r.n}</td>`
        + `<td>${r.pixels.toFixed(4)}</td><td>${r.random.toFixed(4)}</td>`
        + `<td>${r.shuffled.toFixed(4)}</td><td>${r.transfer.toFixed(4)}</td>`
        + `<td>${r.oracle.toFixed(4)}</td></tr>`).join('')
      + `</tbody></table><div class="gen-note">With ${small.n} labelled rows the transferred `
      + `encoder scores ${small.transfer.toFixed(4)} and the untrained one of the same shape `
      + `scores ${small.random.toFixed(4)}, so `
      + `${(small.transfer - small.random).toFixed(4)} of the `
      + `${(small.transfer - small.pixels).toFixed(4)} it has over raw pixels is what the `
      + `training bought and the rest is what the projection bought. `
      + (closes
        ? `Training from scratch catches up at ${closes.n} rows, where the two are within `
          + `their own seed spread of each other.`
        : `Training from scratch has not caught up by ${big.n} rows: it is still `
          + `${(big.transfer - big.scratch).toFixed(4)} behind.`)
      + `</div>`;
  }

  render();
  return { render, state: st };
}
