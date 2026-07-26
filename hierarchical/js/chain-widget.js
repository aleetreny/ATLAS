/* The bridge: single linkage's downfall, live. Two clear blobs, twelve
 * stepping stones between them, and four linkages the reader can swap while
 * sliding k. Single walks the bridge and unites what should be apart; ward
 * pays the variance bill and refuses. All clustering here is this page's own
 * Lance-Williams, checked against scipy at load time.
 */
import { drawGrid, drawAxes, makeChart } from '../../assets/js/chart.js';
import { agglomerate, cutTree } from './hierkit.js';

const COLOURS = ['var(--aqua)', 'var(--cosmos)', 'var(--anchor)', 'var(--smile)', 'var(--galaxy)', '#0e8a55'];

export function initChainWidget(data) {
  const pts = data.chain.points;
  const { g, w, h } = makeChart('#chain-chart', {
    width: 660,
    height: 400,
    margin: { top: 34, right: 26, bottom: 46, left: 50 },
  });
  const x = d3.scaleLinear().domain(d3.extent(pts, (p) => p[0])).nice().range([0, w]);
  const y = d3.scaleLinear().domain(d3.extent(pts, (p) => p[1])).nice().range([h, 0]);
  drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 4 });
  drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 4 });
  const dotG = g.append('g');
  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -14).attr('text-anchor', 'middle').style('font-size', '0.64rem');

  /* All four trees computed once, live. */
  const trees = {};
  for (const m of ['ward', 'complete', 'average', 'single']) trees[m] = agglomerate(pts, m);

  const btnRow = document.querySelector('#chain-buttons');
  const slider = document.querySelector('#chain-k');
  const readout = document.querySelector('#chain-readout');
  const state = { method: 'single' };

  const btns = {};
  for (const m of ['single', 'average', 'complete', 'ward']) {
    const b = document.createElement('button');
    b.className = 'atlas-btn' + (m === 'single' ? '' : ' ghost');
    b.textContent = m;
    b.addEventListener('click', () => { state.method = m; render(); });
    btnRow.appendChild(b);
    btns[m] = b;
  }

  function render() {
    const k = +slider.value;
    document.querySelector('#chain-k-label').textContent = k;
    const labels = cutTree(trees[state.method], pts.length, k);
    dotG.selectAll('circle').data(pts).join('circle')
      .attr('cx', (p) => x(p[0])).attr('cy', (p) => y(p[1]))
      .attr('r', 3.6)
      .attr('fill', (p, i) => COLOURS[labels[i] % COLOURS.length])
      .attr('stroke', 'white').attr('stroke-width', 0.6).attr('opacity', 0.85);
    for (const [m, b] of Object.entries(btns)) b.classList.toggle('ghost', m !== state.method);

    /* Are the two blobs in different clusters at this cut? */
    const majority = (from, to) => {
      const counts = {};
      for (let i = from; i < to; i++) counts[labels[i]] = (counts[labels[i]] || 0) + 1;
      return +Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
    };
    const separated = majority(0, 55) !== majority(55, 110);
    title.text(`${state.method} linkage, cut at k = ${k}: blobs ${separated ? 'separated' : 'CHAINED TOGETHER'}`);

    readout.innerHTML =
      `<div class="slider-label" style="line-height:1.5">${
        state.method === 'single'
          ? separated
            ? `At k = ${k} even single linkage separates the blobs, because enough cuts land on the ` +
              `bridge to break it into pieces first. Slide back to k = 2 for the classic disaster.`
            : `Single linkage asks only "who is my nearest neighbour", and every stepping stone has one ` +
              `about 0.4 away. The bridge becomes a corridor, the corridor unites the blobs, and at ` +
              `k = 2 the split it offers instead is some stray point versus everything else. One dozen ` +
              `points defeated one hundred and ten.`
          : separated
            ? `${state.method === 'ward' ? 'Ward charges every merge by the variance it adds, and ' +
                'absorbing a bridge four units long into a compact blob is exorbitant: the blobs stay ' +
                'apart and the bridge is split between them.' : 'Merging by ' + (state.method === 'complete' ? 'farthest' : 'average') +
                ' pair distance makes the long thin corridor expensive, so the blobs survive.'} ` +
              `The same twelve points that owned single linkage cost the others nothing.`
            : `Merged: at this k the cut is too coarse even for ${state.method}.`
      }</div>`;
  }

  slider.addEventListener('input', render);
  render();
}
