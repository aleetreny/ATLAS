/* The scoreboard: the same 2,000 digits the three previous articles used, mapped
 * five ways, scored the same three ways module 2.2 has been scoring things all
 * along.
 *
 * The two neighbour embeddings are drawn from coordinates the generator
 * computed, and the widget says so on the panel. t-SNE at this size is twenty
 * seconds of arithmetic and UMAP is not implemented here at all: what the
 * browser recomputes elsewhere on this page is the 500 point version, where it
 * can be checked.
 */
import { makeChart, wrapLabel, equalScales } from '../../assets/js/chart.js';
import { toRows } from '../../assets/js/embed.js';

const DIGIT = d3.scaleOrdinal(d3.schemeCategory10).domain(d3.range(10));

export function initBoardWidget(data) {
  const chart = makeChart('#board-chart', {
    width: 780, height: 420, margin: { top: 76, right: 22, bottom: 30, left: 40 },
  });
  const { g, w, h } = chart;
  const buttons = document.querySelector('#board-buttons');
  const readout = document.querySelector('#board-readout');
  const st = { key: 't-sne' };
  /* No PCA panel here: the scoreboard runs on 2,000 digits and the page only
     downloads the 500 it recomputes things on, so a PCA picture at this size
     would have to be shipped too, for a panel whose row is already in the
     table and whose shape the scrolly above already shows. */

  const maps = {
    pca: { coords: null, label: 'PCA', note: '' },
    't-sne': {
      coords: toRows(data.big.tsne),
      label: 't-SNE',
      note: `computed by the generator at ${data.meta.n_big.toLocaleString('en-US')} points`,
    },
    umap: {
      coords: data.umap.big ? toRows(data.umap.big) : null,
      label: 'UMAP',
      note: 'computed by the umap-learn package',
    },
  };

  const btns = {};
  ['t-sne', 'umap'].forEach((key) => {
    if (!maps[key].coords) return;
    const b = document.createElement('button');
    b.className = 'atlas-btn' + (key === st.key ? '' : ' ghost');
    b.textContent = maps[key].label;
    b.addEventListener('click', () => {
      st.key = key;
      render();
    });
    buttons.appendChild(b);
    btns[key] = b;
  });

  const body = g.append('g');
  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -58).attr('text-anchor', 'middle')
    .style('font-size', '12px').style('text-transform', 'none');
  const legend = g.append('g');
  d3.range(10).forEach((k) => {
    legend.append('circle').attr('cx', 8 + k * 74).attr('cy', -22).attr('r', 4.5)
      .attr('fill', DIGIT(k));
    legend.append('text').attr('x', 18 + k * 74).attr('y', -18)
      .style('font-size', '11.5px').style('font-family', 'var(--font-mono)')
      .attr('fill', 'var(--squidink)').text(k);
  });

  function render() {
    for (const [k, b] of Object.entries(btns)) b.classList.toggle('ghost', k !== st.key);
    const M = maps[st.key];
    const Y = M.coords;
    const { x, y } = equalScales(Y, w, h, { pad: 1.08 });
    body.selectAll('circle').data(Y).join('circle')
      .attr('cx', (d) => x(d[0])).attr('cy', (d) => y(d[1])).attr('r', 2.2)
      .attr('fill', (d, i) => DIGIT(data.big.labels[i])).attr('opacity', 0.8);
    const row = data.board.find((r) => r.method === st.key);
    wrapLabel(title, `${M.label} on the same ${data.meta.n_big.toLocaleString('en-US')} digits, `
      + `${M.note}${row && row.note ? `, ${row.note}` : ''}`, w - 8);

    const rows = data.board.map((r) => {
      const mark = r.method === st.key ? ' style="font-weight:700"' : '';
      return `<tr${mark}><td>${maps[r.method] ? maps[r.method].label : r.method}</td>`
        + `<td>${(r.accuracy * 100).toFixed(1)}%</td>`
        + `<td>${r.trustworthiness}</td><td>${r.global_corr}</td></tr>`;
    }).join('');
    const pca = data.board.find((r) => r.method === 'pca');
    const tsne = data.board.find((r) => r.method === 't-sne');
    readout.innerHTML = `
      <table class="nb-table">
        <tr><th>method</th><th>5-nn on the map</th><th>trustworthiness</th><th>agreement with the original distances</th></tr>
        ${rows}
      </table>
      <div class="nb-note">
        The trade is exact and it goes both ways. Two coordinates from
        ${maps['t-sne'].label} classify a digit
        <span class="bold">${(tsne.accuracy / pca.accuracy).toFixed(1)} times</span> better than two
        from PCA (${(tsne.accuracy * 100).toFixed(1)}% against
        ${(pca.accuracy * 100).toFixed(1)}%), and its distances agree with the real ones
        <span class="bold">less</span> well: ${tsne.global_corr} against PCA's
        ${pca.global_corr}. Nothing was cheated. The neighbour embeddings spent everything they had
        on getting neighbourhoods right and paid for it with the only currency available, which is
        what the distances in the picture mean.
      </div>`;
  }

  render();
}
