/* The two metrics the StyleGAN paper introduced, on a dataset where a third
 * one is available that nobody usually has.
 *
 * Perceptual path length asks how far the picture travels for a small step in
 * the latent, averaged over the space: a mapping that has to bend to avoid a
 * hole in the data has to travel fast somewhere. Linear separability asks
 * whether a straight line in the latent space can tell one attribute from
 * another. Both are computed in z and, where there is one, in w. The third is
 * the same path length measured in the TRUE factors of the sprite rather than
 * in a network's features, which is only possible because this data was drawn
 * rather than collected.
 */
import { makeChart, drawAxes, axisLabels } from '../../assets/js/chart.js';

const SPACE = { z: 'the noise, z', w: 'the mapped latent, w' };
const ATTR = { large: 'is it large', blue: 'is it blue', right: 'is it on the right', disc: 'is it a disc' };

export function initMetricsWidget(data) {
  const M = data.meta;
  const seed = M.seeds[0];
  const plain = data.metrics[`plain/${seed}`];
  const styled = data.metrics[`styled/${seed}`];

  const c = makeChart('#ppl-chart', {
    width: 560, height: 320, margin: { top: 40, right: 30, bottom: 70, left: 76 },
  });
  /* Short enough to sit under a bar without colliding with its neighbour: the
     full names are in the table under the chart. */
  const bars = [
    { k: 'input, in z', v: plain.path.z.perceptual, colour: 'var(--anchor)' },
    { k: 'every layer, in z', v: styled.path.z.perceptual, colour: 'var(--primary)' },
    { k: 'every layer, in w', v: styled.path.w.perceptual, colour: 'var(--cosmos)' },
  ];
  const x = d3.scaleBand().domain(bars.map((b) => b.k)).range([0, c.w]).padding(0.3);
  const y = d3.scaleLinear().domain([0, d3.max(bars, (b) => b.v) * 1.18]).range([c.h, 0]);
  drawAxes(c.g, x, y, c.w, c.h, { yTicks: 5 });
  axisLabels(c.g, c.w, c.h, { y: 'path length' });
  c.g.selectAll('rect').data(bars).enter().append('rect')
    .attr('x', (b) => x(b.k)).attr('y', (b) => y(b.v))
    .attr('width', x.bandwidth()).attr('height', (b) => c.h - y(b.v))
    .attr('fill', (b) => b.colour);
  c.g.selectAll('text.val').data(bars).enter().append('text').attr('class', 'chart-annotation val')
    .attr('x', (b) => x(b.k) + x.bandwidth() / 2).attr('y', (b) => y(b.v) - 7)
    .attr('text-anchor', 'middle').style('font-size', '11.5px').style('letter-spacing', '0.4px')
    .text((b) => b.v.toFixed(1));
  c.g.selectAll('.axis.x text').style('font-size', '10px');

  const rows = [];
  [['plain', plain], ['styled', styled]].forEach(([name, m]) => {
    Object.entries(m.path).forEach(([space, p]) => {
      rows.push(`<tr><td>${name === 'plain' ? 'at the input' : 'at every layer'}</td>`
        + `<td>${SPACE[space]}</td><td>${p.perceptual.toFixed(1)}</td>`
        + `<td>${p.factor.toFixed(1)}</td><td>${p.roughness.toFixed(2)}</td></tr>`);
    });
  });
  document.querySelector('#ppl-readout').innerHTML =
    `<table class="atlas-table"><thead><tr><th>Where the latent enters</th><th>Space stepped in</th>`
    + `<th>Path length, in features</th><th>Path length, in true factors</th>`
    + `<th>Roughness</th></tr></thead><tbody>${rows.join('')}</tbody></table>`
    + `<p style="margin-top:0.8rem">Roughness is the ninetieth percentile of the step length over `
    + `its median: how uneven the journey is, rather than how long. A perfectly smooth `
    + `parameterisation would sit near one, and every extra unit is somewhere in the space where a `
    + `small change of latent produces a jump.</p>`;

  /* separability */
  const attrs = Object.keys(ATTR);
  const sepRows = [];
  [['at the input', plain.sep], ['at every layer', styled.sep]].forEach(([name, sp]) => {
    Object.entries(sp).forEach(([space, byAttr]) => {
      const cells = attrs.map((a) => {
        const r = byAttr[a];
        if (!r || r.acc === null) return '<td>not present</td>';
        return `<td>${(r.acc * 100).toFixed(1)}% <span style="opacity:0.6">of `
          + `${(r.base * 100).toFixed(0)}%</span></td>`;
      }).join('');
      sepRows.push(`<tr><td>${name}</td><td>${SPACE[space]}</td>${cells}</tr>`);
    });
  });
  document.querySelector('#sep-readout').innerHTML =
    `<table class="atlas-table"><thead><tr><th>Where the latent enters</th><th>Space</th>`
    + attrs.map((a) => `<th>${ATTR[a]}</th>`).join('')
    + `</tr></thead><tbody>${sepRows.join('')}</tbody></table>`
    + `<p style="margin-top:0.8rem">Each cell is the accuracy of a straight line drawn in that `
    + `space, on held-out samples, with the accuracy of always guessing the commoner answer in `
    + `grey next to it. A cell that matches its grey number means the space carries the attribute `
    + `nowhere a line can find it.</p>`;
}
