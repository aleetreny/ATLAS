/* Ten handwritten digits, placed by nothing but how often a classifier mixes
 * them up. There is no data matrix here to rotate, which is the point: this is
 * the job PCA cannot be asked to do.
 *
 * The dissimilarity comes from the generator (it takes a 2,000 image fit and a
 * 2,000 image test set to make). Everything drawn from it, both maps and the
 * spectrum, is computed here.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel, equalScales } from '../../assets/js/chart.js';
import { classicalMDS, smacof, upperOrder, negativeMass, stress1, toRows } from '../../assets/js/embed.js';

export function digitsSolve(data) {
  const D = toRows(data.digits.dissimilarity);
  const { coords, values } = classicalMDS(D, 2);
  /* A ten point configuration needs a starting point like any other, and this
     one is the generator's, shipped in the file, so the two runs are the same
     deterministic trajectory rather than two things that ought to look alike. */
  const init = toRows(data.digits.init);
  const nm = smacof(D, init, data.digits.smacof_iters, { nonmetric: true, order: upperOrder(D) });
  return {
    D,
    classical: coords,
    nonmetric: nm.coords,
    values,
    negMass: negativeMass(values),
    negCount: Array.from(values).filter((v) => v < -1e-9).length,
    stressClassical: stress1(D, coords),
    stressNonmetric: stress1(nm.target, nm.coords),
  };
}

export function initDigitsWidget(data, solved) {
  const G = data.digits;
  const chart = makeChart('#digits-chart', {
    width: 700, height: 440, margin: { top: 60, right: 26, bottom: 56, left: 62 },
  });
  const { g, w, h } = chart;
  const viewButtons = document.querySelector('#digits-view-buttons');
  const readout = document.querySelector('#digits-readout');
  const state = { view: 'classical' };

  const btns = {};
  for (const [key, text] of [['classical', 'classical scaling'], ['nonmetric', 'non-metric scaling'],
    ['table', 'the table itself']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn' + (key === state.view ? '' : ' ghost');
    b.textContent = text;
    b.addEventListener('click', () => {
      state.view = key;
      render();
    });
    viewButtons.appendChild(b);
    btns[key] = b;
  }

  const body = g.append('g');
  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -42).attr('text-anchor', 'middle')
    .style('font-size', '12px').style('text-transform', 'none');

  function render() {
    for (const [k, b] of Object.entries(btns)) b.classList.toggle('ghost', k !== state.view);
    body.selectAll('*').remove();
    g.selectAll('g.axis, g.grid, text.axis-label').remove();

    if (state.view === 'table') {
      const m = 10;
      const cell = Math.min(34, (h - 30) / m);
      const x0 = (w - cell * m) / 2;
      const y0 = 16;
      for (let i = 0; i < m; i++) {
        body.append('text').attr('x', x0 + cell * (i + 0.5)).attr('y', y0 - 6)
          .attr('text-anchor', 'middle').style('font-size', '12px')
          .style('font-family', 'var(--font-heavy)').attr('fill', 'var(--squidink)').text(i);
        body.append('text').attr('x', x0 - 8).attr('y', y0 + cell * (i + 0.6))
          .attr('text-anchor', 'end').style('font-size', '12px')
          .style('font-family', 'var(--font-heavy)').attr('fill', 'var(--squidink)').text(i);
        for (let j = 0; j < m; j++) {
          const v = solved.D[i][j];
          body.append('rect').attr('x', x0 + cell * j).attr('y', y0 + cell * i)
            .attr('width', cell - 1).attr('height', cell - 1)
            .attr('fill', i === j ? 'var(--paper)'
              : d3.interpolateRgb('#7e22ce', '#ffffff')(v))
            .attr('stroke', 'var(--stone)').attr('stroke-width', 0.5);
          if (i !== j) {
            body.append('text').attr('x', x0 + cell * j + cell / 2)
              .attr('y', y0 + cell * i + cell / 2 + 3).attr('text-anchor', 'middle')
              .style('font-size', '8.5px').style('font-family', 'var(--font-mono)')
              .attr('fill', v < 0.45 ? 'white' : 'var(--squidink)')
              .text(v.toFixed(2));
          }
        }
      }
      body.append('text').attr('x', w / 2).attr('y', y0 + cell * m + 26)
        .attr('text-anchor', 'middle').style('font-size', '11.5px')
        .style('font-family', 'var(--font-mono)').attr('fill', 'var(--squidink)')
        .text(`0 would be identical, 1 is the furthest pair; ${G.never_confused} `
          + 'of the 45 pairs were never confused at all and sit at the maximum');
      wrapLabel(title, 'how far apart two digits are, from how often a classifier swaps them', w - 8);
    } else {
      const Z = state.view === 'classical' ? solved.classical : solved.nonmetric;
      const { x, y } = equalScales(Z, w, h, { pad: 1.24 });
      drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
      drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
      axisLabels(g, w, h, { x: 'first coordinate', y: 'second coordinate' });
      /* The five most confusable pairs, drawn as the ties they are. */
      G.top_pairs.forEach((p) => {
        body.append('line')
          .attr('x1', x(Z[p.a][0])).attr('y1', y(Z[p.a][1]))
          .attr('x2', x(Z[p.b][0])).attr('y2', y(Z[p.b][1]))
          .attr('stroke', 'var(--primary)').attr('stroke-width', 4.5 - 3.2 * p.d)
          .attr('opacity', 0.35);
      });
      Z.forEach((d, i) => {
        body.append('circle').attr('cx', x(d[0])).attr('cy', y(d[1])).attr('r', 17)
          .attr('fill', 'white').attr('stroke', 'var(--primary)').attr('stroke-width', 2);
        body.append('text').attr('x', x(d[0])).attr('y', y(d[1]) + 7)
          .attr('text-anchor', 'middle').style('font-size', '19px')
          .style('font-family', 'var(--font-heavy)').attr('fill', 'var(--squidink)').text(i);
      });
      wrapLabel(title, state.view === 'classical'
        ? 'classical scaling, which takes the numbers at face value'
        : 'non-metric scaling, which takes only their order', w - 8);
    }

    const s = state.view === 'nonmetric' ? solved.stressNonmetric : solved.stressClassical;
    readout.innerHTML = `
      <table class="dist-table">
        <tr><th>classifier</th><th>most confusable pair</th><th>never confused</th><th>negative mass</th><th>stress shown</th></tr>
        <tr>
          <td>${G.k}-nn, ${(G.accuracy * 100).toFixed(1)}% correct</td>
          <td><span class="value">${G.closest[0]} and ${G.closest[1]}</span> at ${G.closest_d}</td>
          <td>${G.never_confused} of 45 pairs</td>
          <td><span class="value">${(solved.negMass * 100).toFixed(2)}%</span> over ${solved.negCount} eigenvalue${solved.negCount === 1 ? '' : 's'}</td>
          <td><span class="value">${s.toFixed(4)}</span></td>
        </tr>
      </table>
      <div class="dist-note">
        ${state.view === 'table'
    ? `Every number here was measured, none of them was placed. The classifier is a plain `
          + `five-nearest-neighbour vote on raw pixels, and the only thing it contributes is a count `
          + `of how often it mistakes one digit for another.`
    : `Stress ${s.toFixed(4)} against ${(state.view === 'classical' ? solved.stressNonmetric
    : solved.stressClassical).toFixed(4)} for the other button. `
          + `${solved.stressNonmetric < solved.stressClassical
    ? `The non-metric fit is the tighter of the two, and it should be: ${G.never_confused} of the 45 `
              + `pairs are tied at the maximum because the classifier never once swapped them, and a `
              + `method that only reads the ordering is allowed to leave a tie as a tie instead of `
              + `insisting all ${G.never_confused} of those pairs are exactly equally far apart.`
    : `Here the classical fit is the tighter one.`}`}
      </div>`;
  }

  render();
}
