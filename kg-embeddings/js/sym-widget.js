/* The closed form prediction, plotted against the trained parameters.
 *
 * TransE says a symmetric relation forces its vector to zero. RotatE says the
 * same relation forces its rotation to be its own inverse, which means an
 * angle of zero or pi. Both are statements about numbers inside a trained
 * model, and both can be checked by measuring how symmetric each relation
 * actually is in the data and putting the two on the same axes.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, spread } from '../../assets/js/chart.js';

export function initSymWidget(data) {
  const rels = data.relations;
  const buttons = document.querySelector('#sym-buttons');
  const readout = document.querySelector('#sym-readout');
  const st = { view: 'transe' };

  const chart = makeChart('#sym-chart', {
    width: 640, height: 400, margin: { top: 44, right: 128, bottom: 60, left: 76 }, clip: true,
  });
  const { g, plot, w, h, clear } = chart;

  const btns = {};
  for (const [key, text] of [['transe', 'TransE: the length of the vector'],
    ['rotate', 'RotatE: how close to its own inverse']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', () => { st.view = key; render(); });
    buttons.appendChild(b);
    btns[key] = b;
  }

  const nice = (n) => n.replace(/^_/, '').replace(/_/g, ' ');

  function render() {
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', k !== st.view));
    clear();
    g.selectAll('text.chart-note').remove();

    const model = data.wn[st.view];
    const pts = rels.map((r) => {
      const m = model.relations[String(r.id)];
      return {
        name: nice(r.name), n: r.n, sym: r.symmetry,
        v: st.view === 'transe' ? m.norm : m.self_inverse,
      };
    });

    const x = d3.scaleLinear().domain([-0.04, 1.04]).range([0, w]);
    const y = d3.scaleLinear().domain([0, d3.max(pts, (p) => p.v) * 1.18]).range([h, 0]);
    drawGrid(g, x, y, w, h, { xTicks: 6, yTicks: 5 });

    const size = d3.scaleSqrt().domain([0, d3.max(pts, (p) => p.n)]).range([3, 11]);
    plot.selectAll(null).data(pts).enter().append('circle')
      .attr('cx', (p) => x(p.sym)).attr('cy', (p) => y(p.v)).attr('r', (p) => size(p.n))
      .attr('fill', (p) => (p.sym > 0.5 ? 'var(--primary)' : 'var(--anchor)'))
      .attr('opacity', 0.85);

    /* only the relations at the two ends get a label, or eleven names in a
       narrow panel become one grey smear */
    const labelled = pts.filter((p) => p.sym > 0.5 || p.n > 4000);
    const placed = spread(labelled.map((p) => ({ p, y: y(p.v) + 3.5, yMax: h })), 13);
    placed.forEach(({ p, y: yy }) => {
      const left = x(p.sym) > w * 0.6;
      g.append('text').attr('class', 'chart-note')
        .attr('x', x(p.sym) + (left ? -10 : 10)).attr('y', yy)
        .attr('text-anchor', left ? 'end' : 'start')
        .style('font-size', '10.5px').text(p.name);
    });

    drawAxes(g, x, y, w, h, { xTicks: 6, yTicks: 5, xFmt: d3.format('.0%'),
      yFmt: d3.format(st.view === 'transe' ? '.1f' : '.2f') });
    axisLabels(g, w, h, {
      x: 'share of this relation that is symmetric in the data',
      y: st.view === 'transe' ? 'length of the learned relation vector'
        : 'distance from a self inverse rotation',
    });
    g.append('text').attr('class', 'chart-note')
      .attr('x', w / 2).attr('y', -24).attr('text-anchor', 'middle').style('font-size', '12px')
      .text(`the ${rels.length} relations of WN18RR, sized by how many triples they have`);

    const symmetric = pts.filter((p) => p.sym > 0.5);
    const anti = pts.filter((p) => p.sym <= 0.5);
    const meanSym = symmetric.reduce((a, p) => a + p.v, 0) / Math.max(symmetric.length, 1);
    const meanAnti = anti.reduce((a, p) => a + p.v, 0) / Math.max(anti.length, 1);
    readout.innerHTML = st.view === 'transe'
      ? `The ${symmetric.length} symmetric relations have relation vectors of average length `
        + `<span class="value">${meanSym.toFixed(3)}</span>, and the ${anti.length} that are not `
        + `symmetric average <span class="value">${meanAnti.toFixed(3)}</span>, a factor of `
        + `<span class="value">${(meanAnti / Math.max(meanSym, 1e-9)).toFixed(1)}</span>. `
        /* the sentence after the ratio has to be decided by the ratio, not by
           the argument the section was written to make */
        + `${meanSym < meanAnti
          ? 'That is the algebra showing up in the parameters: if h + r has to be near t and t + r '
            + 'near h, then r has nowhere to go but zero, and a relation vector of zero says that '
            + 'every pair it connects should be at the same point.'
          : 'Which is the wrong way round for the algebra: if h + r has to be near t and t + r '
            + 'near h then r has nowhere to go but zero, and this optimum did not go there. The '
            + 'proof is about where the minimum is, not about where a finite budget stops.'}`
      : `The same relations under RotatE, measured by how far each rotation is from being its own `
        + `inverse, which is what symmetry demands. The ${symmetric.length} symmetric relations `
        + `average <span class="value">${meanSym.toFixed(3)}</span> and the ${anti.length} others `
        + `<span class="value">${meanAnti.toFixed(3)}</span>. `
        + `${meanSym < meanAnti
          ? 'The symmetric ones sit closer to a half turn or no turn at all, which is the only '
            + 'thing a rotation can be if applying it twice has to bring you home. Nothing here '
            + 'forced that: it came out of the fit.'
          : 'The separation is not there in this run, which is worth reporting rather than tuning '
            + 'away: the prediction is about the optimum, and this is what the optimiser reached.'}`;
  }

  render();
  return { render, state: st };
}
