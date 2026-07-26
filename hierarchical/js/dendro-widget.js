/* The whole point of the article in one object: all 178 wines as a tree, and
 * a blade the reader drags. The cut height IS the clustering: drag it down
 * and clusters split, drag it up and they merge, and the PCA scatter plus a
 * live ARI against the never-shown cultivars follow every move. The four
 * linkage buttons swap in scipy's tree for each method, including single
 * linkage's infamous comb.
 */
import { makeChart } from '../../assets/js/chart.js';
import { cutTree, dendrogram, ari } from './hierkit.js';

const COLOURS = ['var(--aqua)', 'var(--cosmos)', 'var(--anchor)', 'var(--smile)',
  'var(--galaxy)', '#0e8a55', '#7c5aed', '#a8410f'];
const N = 178;

export function initDendroWidget(data) {
  const W = data.wine;

  /* Top: the tree. Bottom-left: the PCA projection. Bottom-right: readout
   * handled by the shared .widget-readout div. */
  const { g, w, h } = makeChart('#dendro-chart', {
    width: 700,
    height: 380,
    margin: { top: 26, right: 20, bottom: 26, left: 52 },
  });
  const P = makeChart('#dendro-proj', {
    width: 700,
    height: 320,
    margin: { top: 24, right: 20, bottom: 34, left: 52 },
  });
  const px = d3.scaleLinear().domain(d3.extent(W.proj, (p) => p[0])).nice().range([0, P.w]);
  const py = d3.scaleLinear().domain(d3.extent(W.proj, (p) => p[1])).nice().range([P.h, 0]);
  const projG = P.g.append('g');
  const pTitle = P.g.append('text').attr('class', 'chart-annotation')
    .attr('x', P.w / 2).attr('y', -8).attr('text-anchor', 'middle').style('font-size', '0.6rem');

  const treeG = g.append('g');
  const axisG = g.append('g');
  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -10).attr('text-anchor', 'middle').style('font-size', '0.62rem');

  const readout = document.querySelector('#dendro-readout');
  const btnRow = document.querySelector('#dendro-buttons');
  const state = { method: 'ward', cutFrac: null };

  const btns = {};
  for (const m of ['ward', 'complete', 'average', 'single']) {
    const b = document.createElement('button');
    b.className = 'atlas-btn' + (m === 'ward' ? '' : ' ghost');
    b.textContent = m;
    b.addEventListener('click', () => {
      state.method = m;
      state.cutFrac = null;   // each tree gets a fresh, sensible starting cut
      render();
    });
    btnRow.appendChild(b);
    btns[m] = b;
  }

  let drag = null;

  function render() {
    const L = W.linkages[state.method];
    const Z = L.Z;
    const layout = dendrogram(Z, N);
    const maxH = Z[Z.length - 1][2];
    const x = d3.scaleLinear().domain([0, N - 1]).range([0, w]);
    const y = d3.scaleSqrt().domain([0, maxH * 1.04]).range([h, 0]);
    axisG.selectAll('*').remove();
    axisG.call(d3.axisLeft(y).ticks(5));
    axisG.selectAll('text').style('font-family', 'var(--font-mono)').style('font-size', '10px');

    /* Default cut: halfway between the merge that leaves 3 clusters (row
     * N-4, from 4 to 3) and the one that ends them (row N-3, from 3 to 2),
     * so every tree opens on k = 3 and the k-means comparison is immediate. */
    if (state.cutFrac === null) {
      state.cutFrac = (Z[N - 4][2] + Z[N - 3][2]) / 2 / maxH;
    }
    const cutH = state.cutFrac * maxH;
    const k = Z.filter((row) => row[2] >= cutH).length + 1;
    const labels = cutTree(Z, N, k);
    const a = ari(labels, W.true);

    /* Colour every merge below the cut by its cluster; grey above. */
    const leafPos = new Map(layout.leaves.map((v, i) => [v, i]));
    const leafOf = new Array(2 * N - 1);
    layout.leaves.forEach((v) => { leafOf[v] = v; });
    Z.forEach((row, s) => { leafOf[N + s] = leafOf[row[0]]; });

    treeG.selectAll('path').data(layout.merges).join('path')
      .attr('fill', 'none')
      .attr('stroke', (m, s) => (Z[s][2] < cutH ? COLOURS[labels[leafOf[N + s]] % COLOURS.length] : 'var(--stone)'))
      .attr('stroke-width', (m, s) => (Z[s][2] < cutH ? 1.6 : 1.2))
      .attr('d', (m) =>
        `M${x(m.xa)},${y(m.ha)} L${x(m.xa)},${y(m.h)} L${x(m.xb)},${y(m.h)} L${x(m.xb)},${y(m.hb)}`);

    /* The blade. */
    if (!drag) {
      drag = g.append('g').style('cursor', 'ns-resize');
      drag.append('line').attr('x1', 0)
        .attr('stroke', 'var(--squidink)').attr('stroke-width', 2.4).attr('stroke-dasharray', '8 5');
      drag.append('rect').attr('x', 0).attr('y', -11).attr('height', 22).attr('fill', 'transparent');
      drag.append('text').attr('class', 'chart-annotation')
        .attr('text-anchor', 'end').style('font-size', '0.62rem').style('text-transform', 'none')
        .attr('fill', 'var(--squidink)');
      drag.call(d3.drag().on('drag', (event) => {
        state.cutFrac = Math.max(0.001, Math.min(y.invert(Math.max(2, Math.min(event.y, h - 2))) / maxH, 0.999));
        render();
      }));
    }
    drag.attr('transform', `translate(0,${y(cutH)})`);
    drag.select('line').attr('x2', w);
    drag.select('rect').attr('width', w);
    drag.select('text').attr('x', w - 4).attr('y', -6)
      .text(`the cut (drag me): k = ${k}`);

    projG.selectAll('circle').data(W.proj).join('circle')
      .attr('cx', (p) => px(p[0])).attr('cy', (p) => py(p[1]))
      .attr('r', 3.6)
      .attr('fill', (p, i) => COLOURS[labels[i] % COLOURS.length])
      .attr('stroke', 'white').attr('stroke-width', 0.6).attr('opacity', 0.85);
    pTitle.text(`the same cut, seen in the projection: k = ${k}, ARI ${a.toFixed(3)} against the cultivars`);

    for (const [m, b] of Object.entries(btns)) b.classList.toggle('ghost', m !== state.method);
    title.text(`${state.method} linkage: 178 bottles, 177 merges, one tree`);

    const rows = Object.entries(L.ari_by_k).map(([kk, v]) =>
      `<td style="${+kk === Math.min(k, 8) ? 'font-weight:700' : ''}">${v.toFixed(3)}</td>`).join('');
    readout.innerHTML =
      `<table class="hier-table">
         <tr><th>at this cut</th><th>k</th><th>ARI</th></tr>
         <tr><td>${state.method}</td><td><span class="value">${k}</span></td><td><span class="value">${a.toFixed(3)}</span></td></tr>
       </table>
       <table class="hier-table" style="margin-top:.4rem">
         <tr><th>k</th>${Object.keys(L.ari_by_k).map((kk) => `<th>${kk}</th>`).join('')}</tr>
         <tr><td>ARI</td>${rows}</tr>
       </table>` +
      `<div class="slider-label" style="margin-top:.6rem;line-height:1.5">${
        state.method === 'single'
          ? `The comb. Single linkage merges by nearest neighbour, so one bottle at a time joins one ` +
            `giant cluster, and no cut anywhere in this tree separates the cultivars (best ARI ` +
            `${Math.max(...Object.values(L.ari_by_k)).toFixed(3)}). The tree shape itself is the ` +
            `diagnosis: combs mean chaining.`
          : state.method === 'ward'
            ? `Ward merges whichever pair least increases within-cluster variance, the same objective ` +
              `k-means descends. Drag the blade through the tall three-branch storey and k = 3 appears ` +
              `with ARI ${L.ari_by_k['3'].toFixed(3)}: a step behind k-means' flat 0.897, and the gap is ` +
              `the price of greed. Every merge is forever, so one early mistake rides all the way up the ` +
              `tree. What the fee buys is everything above the blade: the whole nested story, at every ` +
              `resolution at once, and the tall clear gaps that say "this split is real".`
            : `${L.ari_by_k['3'] < 0.1
                ? `Look at the first cuts: ARI ${L.ari_by_k['3'].toFixed(3)} at k = 3, because this ` +
                  `linkage merges outliers last, so the early cuts just peel off single bottles while ` +
                  `the real mass stays fused. Only by k = ${L.best_k} (ARI ` +
                  `${L.ari_by_k[String(L.best_k)].toFixed(3)}) has it shed enough strays to say anything.`
                : `A compromise linkage: better than single's comb, behind ward here (ARI ` +
                  `${L.ari_by_k['3'].toFixed(3)} at k = 3 against ward's ` +
                  `${W.linkages.ward.ari_by_k['3'].toFixed(3)}), best at k = ${L.best_k} with ` +
                  `${L.ari_by_k[String(L.best_k)].toFixed(3)}.`
              } The lesson of the four buttons is that the linkage choice moves the score far more than ` +
              `any cut does.`
      }</div>`;
  }

  render();
}
