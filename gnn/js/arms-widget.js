/* Every arm on one pair of axes, including the two that are not models.
 *
 * The bar that matters is not the tallest one. Label propagation uses no
 * features at all and the perceptron uses no edges at all, and the distance
 * between those two says which of the two inputs this dataset is actually
 * carrying. The second view is the same GCN with three different
 * normalisations, which is not an architecture choice and moves the number
 * more than switching architecture does.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

const ARMS = [
  ['majority', 'the commonest class', 'var(--stone)'],
  ['mlp', 'features, no edges', 'var(--anchor)'],
  ['label_prop', 'edges, no features', 'var(--cosmos)'],
  ['gcn', 'GCN', 'var(--primary)'],
  ['sage', 'GraphSAGE', 'var(--primary)'],
  ['gat', 'GAT', 'var(--primary)'],
];
const NORMS = [['sym', 'symmetric'], ['row', 'row averaged'], ['none', 'no normalisation']];

export function initArmsWidget(data) {
  const C = data.cora;
  const buttons = document.querySelector('#arms-buttons');
  const readout = document.querySelector('#arms-readout');
  const st = { view: 'arms' };

  const chart = makeChart('#arms-chart', {
    width: 640, height: 390, margin: { top: 44, right: 28, bottom: 78, left: 74 }, clip: true,
  });
  const { g, plot, w, h, clear } = chart;

  const btns = {};
  for (const [key, text] of [['arms', 'the arms'], ['norms', 'the same GCN, three normalisations']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', () => { st.view = key; render(); });
    buttons.appendChild(b);
    btns[key] = b;
  }

  const pct = (v, d = 1) => `${(100 * v).toFixed(d)}%`;

  function render() {
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', k !== st.view));
    clear();
    g.selectAll('text.chart-note').remove();

    const rows = st.view === 'arms'
      ? ARMS.map(([key, label, colour]) => ({ label, colour, v: C[key].acc, sd: C[key].sd }))
      : NORMS.map(([key, label]) => ({ label, colour: 'var(--primary)',
        v: data.norms[key].acc, sd: data.norms[key].sd }));

    const x = d3.scaleBand().domain(rows.map((r) => r.label)).range([0, w]).padding(0.3);
    const y = d3.scaleLinear().domain([0, d3.max(rows, (r) => r.v) * 1.2]).range([h, 0]);
    drawGrid(g, x, y, w, h, { xValues: [], yTicks: 5 });

    plot.selectAll(null).data(rows).enter().append('rect')
      .attr('x', (d) => x(d.label)).attr('width', x.bandwidth())
      .attr('y', (d) => y(d.v)).attr('height', (d) => h - y(d.v))
      .attr('fill', (d) => d.colour);
    rows.forEach((d) => {
      if (d.sd > 0) {
        plot.append('line').attr('x1', x(d.label) + x.bandwidth() / 2)
          .attr('x2', x(d.label) + x.bandwidth() / 2)
          .attr('y1', y(d.v - d.sd)).attr('y2', y(d.v + d.sd))
          .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.4);
      }
      g.append('text').attr('class', 'chart-note')
        .attr('x', x(d.label) + x.bandwidth() / 2).attr('y', y(d.v) - 10)
        .attr('text-anchor', 'middle').style('font-size', '11.5px').text(pct(d.v));
      const words = d.label.split(' ');
      const lines = [];
      let cur = '';
      words.forEach((word) => {
        if ((cur + ' ' + word).trim().length > 12) { lines.push(cur.trim()); cur = word; }
        else cur += ` ${word}`;
      });
      lines.push(cur.trim());
      lines.forEach((ln, i) => {
        g.append('text').attr('class', 'chart-note')
          .attr('x', x(d.label) + x.bandwidth() / 2).attr('y', h + 18 + i * 18)
          .attr('text-anchor', 'middle').style('font-size', '10.5px').text(ln);
      });
    });

    drawAxes(g, x, y, w, h, { xValues: [], yTicks: 5, yFmt: d3.format('.0%') });
    axisLabels(g, w, h, { y: 'test nodes classified correctly' });
    g.append('text').attr('class', 'chart-note')
      .attr('x', w / 2).attr('y', -24).attr('text-anchor', 'middle').style('font-size', '12px')
      .text(st.view === 'arms'
        ? `${data.meta.train} labelled nodes, ${data.meta.test.toLocaleString('en-US')} tested, `
          + `${data.meta.seeds} seeds each`
        : 'the same model and the same seeds, with the mixing matrix built three ways');

    readout.innerHTML = st.view === 'arms'
      ? `The two controls first. A perceptron on the word vectors, with the citations thrown away, `
        + `gets <span class="value">${pct(C.mlp.acc)}</span>. Propagating the `
        + `${data.meta.train} labels along the citations with no features at all gets `
        + `<span class="value">${pct(C.label_prop.acc)}</span>. On this dataset the graph alone is `
        + `${C.label_prop.acc > C.mlp.acc ? 'worth more than' : 'worth less than'} the features `
        + `alone, which is worth knowing before crediting an architecture with anything. Using both, `
        + `a two layer GCN reaches <span class="value">${pct(C.gcn.acc)}</span> `
        + `(&plusmn;${(100 * C.gcn.sd).toFixed(1)} over ${data.meta.seeds} seeds), GraphSAGE `
        + `<span class="value">${pct(C.sage.acc)}</span> and GAT `
        + `<span class="value">${pct(C.gat.acc)}</span>.`
      : `The mixing matrix is not learned and not an architecture: it is a decision made before the `
        + `first weight exists. Symmetric normalisation scores `
        + `<span class="value">${pct(data.norms.sym.acc)}</span>, plain neighbour averaging `
        + `<span class="value">${pct(data.norms.row.acc)}</span>, and summing the neighbours without `
        + `dividing by anything <span class="value">${pct(data.norms.none.acc)}</span>. `
        + `${data.norms.sym.acc - data.norms.none.acc > Math.abs(C.gcn.acc - C.gat.acc)
          ? 'The spread between those three is larger than the spread between the three '
            + 'architectures above.'
          : 'The spread between those three is smaller than the gap between the architectures, '
            + 'which is the comfortable case and not the usual one.'}`;
  }

  render();
  return { render, state: st };
}
