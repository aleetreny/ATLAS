/* The unpaired half: whether the mapping is the right one, and what the cycle
 * is actually made of.
 *
 * Two measurements, both of which need ground truth that a real unpaired
 * dataset does not have. The first is whether the learned correspondence is
 * the correct one, which can be counted here because every plan carries the
 * building type that decides its render. The second is what happens to the
 * reconstruction when the intermediate image is disturbed by an amount too
 * small to see, next to the same disturbance applied to a real image of that
 * domain: if the round trip is carried by a signal at that amplitude, the two
 * numbers come apart.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initCycleWidget(data) {
  const seeds = data.meta.cycle_seeds;
  const rows = seeds.map((s) => {
    const r = data.cycle[String(s)];
    return `<tr><td>seed ${s}</td><td>${(r.agree * 100).toFixed(1)}%</td>`
      + `<td>${r.base_cycle.toFixed(4)}</td>`
      + `<td>${r.agree > 0.75 ? 'the correspondence in the data'
        : r.agree < 0.25 ? 'the mirrored one' : 'neither, consistently'}</td></tr>`;
  }).join('');
  document.querySelector('#cycle-readout').innerHTML =
    `<table class="atlas-table"><thead><tr><th>Run</th><th>Building type preserved</th>`
    + `<th>Round trip error</th><th>What it settled on</th></tr></thead>`
    + `<tbody>${rows}</tbody></table>`;

  /* the perturbation experiment */
  const c = makeChart('#steg-chart', {
    width: 620, height: 340, margin: { top: 36, right: 118, bottom: 56, left: 76 },
  });
  const all = seeds.flatMap((s) => data.cycle[String(s)].noise);
  const x = d3.scaleLinear().domain([0, d3.max(all, (r) => r.sigma) * 1.05]).range([0, c.w]);
  const y = d3.scaleLinear().domain([0, d3.max(all, (r) => Math.max(r.moved_fake, r.moved_real)) * 1.15])
    .range([c.h, 0]);
  drawGrid(c.g, x, y, c.w, c.h, { xTicks: 5, yTicks: 5 });
  drawAxes(c.g, x, y, c.w, c.h, { xTicks: 5, yTicks: 5 });
  axisLabels(c.g, c.w, c.h, {
    x: 'noise added to the image being translated back',
    y: 'how far its translation moved',
  });
  const first = data.cycle[String(seeds[0])];
  [['moved_fake', 'var(--cosmos)', 'a generated image'],
    ['moved_real', 'var(--anchor)', 'a real image']].forEach(([k, colour, label], i) => {
    seeds.forEach((s, si) => {
      const r = data.cycle[String(s)].noise;
      c.g.append('path').attr('fill', 'none').attr('stroke', colour)
        .attr('stroke-width', si === 0 ? 2.4 : 1.2).attr('stroke-opacity', si === 0 ? 1 : 0.55)
        .attr('d', d3.line().x((p) => x(p.sigma)).y((p) => y(p[k]))(r));
    });
    c.g.append('text').attr('class', 'chart-annotation').attr('x', c.w + 6)
      .attr('y', y(first.noise[first.noise.length - 1][k]) + 4 + i * 0)
      .style('font-size', '11.5px').style('letter-spacing', '0.4px').style('fill', colour)
      .text(label);
  });

  const qRows = seeds.map((s) => {
    const r = data.cycle[String(s)];
    return `<tr><td>seed ${s}</td><td>${r.base_cycle.toFixed(4)}</td>`
      + r.quant.map((q) => `<td>${q.cycle.toFixed(4)}</td>`).join('') + '</tr>';
  }).join('');
  document.querySelector('#steg-readout').innerHTML =
    `<p>The dashed pair above is the same noise applied to a real image of the same domain, `
    + `which is the control that makes the comparison mean anything: a translator that is simply `
    + `sensitive would move about as much in both cases.</p>`
    + `<table class="atlas-table"><thead><tr><th>Run</th><th>Round trip, untouched</th>`
    + first.quant.map((q) => `<th>rounded to ${q.bits} bits</th>`).join('')
    + `</tr></thead><tbody>${qRows}</tbody></table>`;
}
