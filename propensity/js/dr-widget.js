/* The two by two table, as a two by two table.
 *
 * Drawn rather than tabulated because the argument is about a pattern across
 * cells, not about four numbers: three cells hold and one does not, and that
 * shape is the whole claim behind the word "doubly".
 */
import { makeChart, annotate } from '../../assets/js/chart.js';

const sgn = (v) => (v >= 0 ? `+${v.toFixed(4)}` : v.toFixed(4));

export function initDrWidget(data) {
  const host = document.querySelector('#dr-widget');
  if (!host) return;
  const D = data.double_robust;
  const chart = makeChart('#dr-chart', {
    width: 640, height: 420, margin: { top: 46, right: 24, bottom: 40, left: 96 },
  });
  const { g, w, h } = chart;
  const readout = document.querySelector('#dr-readout');
  const views = d3.select('#dr-views');
  let method = 'aipw';

  const buttons = [
    ['regression', 'outcome regression'],
    ['ipw', 'weighting'],
    ['aipw', 'doubly robust'],
  ];
  views.selectAll('button')
    .data(buttons)
    .join('button')
    .attr('class', (d) => (d[0] === method ? 'atlas-btn' : 'atlas-btn ghost'))
    .text((d) => d[1])
    .on('click', (_, d) => { method = d[0]; render(); });

  /* key "ab": a = propensity model right, b = outcome model right */
  const cells = [
    { key: '11', row: 0, col: 0 },
    { key: '10', row: 0, col: 1 },
    { key: '01', row: 1, col: 0 },
    { key: '00', row: 1, col: 1 },
  ];
  const biggest = Math.max(...cells.map((c) => Math.abs(D.cells[c.key].aipw.bias)),
    ...cells.map((c) => Math.abs(D.cells[c.key].regression.bias)),
    ...cells.map((c) => Math.abs(D.cells[c.key].ipw.bias)));

  function render() {
    views.selectAll('button')
      .attr('class', (d) => (d[0] === method ? 'atlas-btn' : 'atlas-btn ghost'));
    g.selectAll('*').remove();
    const cw = w / 2;
    const ch = h / 2;
    cells.forEach((c) => {
      const v = D.cells[c.key][method];
      const bad = Math.abs(v.bias) > 4 * Math.abs(v.se);
      const x0 = c.col * cw;
      const y0 = c.row * ch;
      g.append('rect')
        .attr('x', x0 + 5).attr('y', y0 + 5)
        .attr('width', cw - 10).attr('height', ch - 10)
        .attr('fill', bad ? 'var(--cosmos)' : 'var(--primary)')
        .attr('fill-opacity', bad
          ? 0.15 + 0.7 * (Math.abs(v.bias) / biggest)
          : 0.16)
        .attr('stroke', 'var(--squidink)')
        .attr('stroke-width', 1.2);
      g.append('text')
        .attr('x', x0 + cw / 2).attr('y', y0 + ch / 2 - 4)
        .attr('text-anchor', 'middle')
        .style('font-size', '22px')
        .style('font-weight', '700')
        .attr('fill', 'var(--squidink)')
        .text(sgn(v.bias));
      g.append('text')
        .attr('class', 'chart-note')
        .attr('x', x0 + cw / 2).attr('y', y0 + ch / 2 + 20)
        .attr('text-anchor', 'middle')
        .style('font-size', '12px')
        .text(bad ? 'biased' : 'within noise of the truth');
      g.append('text')
        .attr('class', 'chart-note')
        .attr('x', x0 + cw / 2).attr('y', y0 + ch / 2 + 38)
        .attr('text-anchor', 'middle')
        .style('font-size', '11.5px')
        .text(`spread between samples ${v.sd.toFixed(3)}`);
    });

    annotate(g, cw / 2, -22, 'outcome model right', { anchor: 'middle' })
      .style('font-size', '12.5px');
    annotate(g, cw + cw / 2, -22, 'outcome model wrong', { anchor: 'middle' })
      .style('font-size', '12.5px');
    [['propensity right', ch / 2], ['propensity wrong', ch + ch / 2]].forEach(([t, yy]) => {
      g.append('text')
        .attr('class', 'chart-annotation')
        .attr('transform', `rotate(-90) translate(${-yy}, -14)`)
        .attr('text-anchor', 'middle')
        .style('font-size', '12.5px')
        .text(t);
    });

    const label = buttons.find((b) => b[0] === method)[1];
    const cellText = {
      regression: `The outcome regression only reads the top row: it never touches the `
        + `propensity model, so the two cells on the left are its whole story. Where the `
        + `outcome model is right it is unbiased (${sgn(D.cells['11'].regression.bias)} and `
        + `${sgn(D.cells['01'].regression.bias)}), and where it is wrong it is off by `
        + `${sgn(D.cells['10'].regression.bias)} whatever the propensity model does.`,
      ipw: `Weighting is the mirror image: it reads the left column. With the propensity `
        + `model right it is unbiased in both cells (${sgn(D.cells['11'].ipw.bias)} and `
        + `${sgn(D.cells['10'].ipw.bias)}); with it wrong, ${sgn(D.cells['01'].ipw.bias)}, `
        + `and no amount of care with the outcome model rescues it.`,
      aipw: `The doubly robust estimator is the one that reads both, and three of the four `
        + `cells come out inside noise: ${sgn(D.cells['11'].aipw.bias)}, `
        + `${sgn(D.cells['10'].aipw.bias)} and ${sgn(D.cells['01'].aipw.bias)}. The fourth `
        + `is ${sgn(D.cells['00'].aipw.bias)}, which is what doing nothing at all costs. `
        + `"Two chances" is exactly the claim, and the picture is the reason to read the `
        + `name literally.`,
    };
    readout.innerHTML =
      `<p>Each cell is ${D.reps} independent samples of ${D.n} rows, with `
      + `<span class="bold">${label}</span> as the estimator. A cell is called biased when `
      + `its bias is more than four of its own standard errors, which is a threshold on `
      + `something the table already measures rather than a number chosen for the picture. `
      + `${cellText[method]}</p>`;
  }

  render();
}
