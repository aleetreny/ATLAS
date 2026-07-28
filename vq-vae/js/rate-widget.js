/* Rate against distortion, and the only fair way to read the two curves.
 *
 * Both arms spend exactly the same number of bits. One spends them on the pair
 * of coordinates at once (a codebook of 2^b vectors), the other one coordinate
 * at a time, with the split between the axes chosen per seed so the baseline is
 * the best that seed could have built rather than a straw man. Read downwards
 * and one curve is under the other. Read sideways and the gap has units: how
 * many extra bits the separable quantiser needs to reach the same error.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';

export function initRateWidget(data) {
  const rows = data.toy.rows;
  const extra = data.toy.extra_bits;
  const chart = makeChart('#rate-chart', {
    width: 620, height: 360, margin: { top: 52, right: 26, bottom: 56, left: 74 },
  });
  const { g, w, h } = chart;
  const slider = document.querySelector('#rate-slider');
  const valueTag = document.querySelector('#rate-value');
  const readout = document.querySelector('#rate-readout');
  slider.min = 0;
  slider.max = rows.length - 1;
  slider.value = Math.min(5, rows.length - 1);

  const layer = g.append('g');
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -32).attr('text-anchor', 'middle')
    .style('font-size', '12px');

  const bits = rows.map((r) => r.bits);
  const all = rows.flatMap((r) => [r.vector_test.mean, r.scalar_test.mean]);
  const x = d3.scaleLinear().domain([bits[0] - 0.4, bits[bits.length - 1] + 0.4]).range([0, w]);
  const y = d3.scaleLog().domain([Math.min(...all) * 0.7, Math.max(...all) * 1.4]).range([h, 0]);

  function render() {
    const i = +slider.value;
    const row = rows[i];
    valueTag.textContent = `${row.bits} bit${row.bits === 1 ? '' : 's'}, a codebook of ${row.k}`;

    const decades = [1e-3, 1e-2, 1e-1, 1, 10]
      .filter((v) => v >= y.domain()[0] && v <= y.domain()[1]);
    drawGrid(g, x, y, w, h, { xValues: bits, yValues: decades });
    drawAxes(g, x, y, w, h, { xValues: bits, yValues: decades, yFmt: d3.format('.0e') });
    axisLabels(g, w, h, { x: 'bits spent on the code', y: 'squared error, held out' });

    layer.selectAll('*').remove();
    const series = [
      ['one codebook over the pair', (r) => r.vector_test.mean, 'var(--primary)'],
      ['the same bits one axis at a time', (r) => r.scalar_test.mean, 'var(--anchor)'],
    ];
    series.forEach(([label, pick, colour], si) => {
      layer.append('path')
        .attr('d', d3.line().x((r) => x(r.bits)).y((r) => y(pick(r)))(rows))
        .attr('fill', 'none').attr('stroke', colour).attr('stroke-width', 2);
      layer.selectAll(`circle.s${si}`).data(rows).join('circle')
        .attr('class', `s${si}`)
        .attr('cx', (r) => x(r.bits)).attr('cy', (r) => y(pick(r)))
        .attr('r', (r, j) => (j === i ? 6.5 : 3.2))
        .attr('fill', colour);
      /* the two curves descend to the right, so the free space is at the top
         right, and putting the names at the left end put one of them exactly
         where the overhead bracket is drawn */
      layer.append('text').attr('class', 'chart-note')
        .attr('x', w - 4).attr('y', 12 + si * 16).attr('text-anchor', 'end')
        .style('font-size', '11.5px').attr('fill', colour).text(label);
    });

    /* the sideways reading: how far right the other curve has to go */
    const ex = extra.find((e) => e.bits === row.bits);
    if (ex && ex.scalar_bits_needed <= bits[bits.length - 1]) {
      layer.append('line')
        .attr('x1', x(row.bits)).attr('x2', x(ex.scalar_bits_needed))
        .attr('y1', y(row.vector_test.mean)).attr('y2', y(row.vector_test.mean))
        .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.4)
        .attr('stroke-dasharray', '4 3');
      layer.append('text').attr('class', 'chart-note')
        .attr('x', x((row.bits + ex.scalar_bits_needed) / 2))
        .attr('y', y(row.vector_test.mean) - 8).attr('text-anchor', 'middle')
        .style('font-size', '11.5px')
        .text(`${ex.extra} bits of overhead`);
    }
    wrapLabel(title, 'the same error, reached at two different prices', w - 8);

    readout.innerHTML = `
      <table class="gen-table">
        <tr>
          <th>bits</th><th>entries</th><th>one codebook over the pair</th>
          <th>the same bits one axis at a time</th><th>the split it chose</th>
          <th>what the alphabet is really worth</th>
        </tr>
        <tr>
          <td><span class="value">${row.bits}</span></td>
          <td>${row.k}</td>
          <td><span class="value">${row.vector_test.mean}</span> &plusmn; ${row.vector_test.spread}</td>
          <td><span class="value">${row.scalar_test.mean}</span> &plusmn; ${row.scalar_test.spread}</td>
          <td>${row.scalar_split[0]} and ${row.scalar_split[1]} bits</td>
          <td>${row.usage_bits.mean} bits of entropy, of ${row.bits}</td>
        </tr>
      </table>
      <div class="gen-note">
        At ${row.bits} bit${row.bits === 1 ? '' : 's'} the codebook reaches ${row.vector_test.mean}
        and spending the same bits one coordinate at a time reaches ${row.scalar_test.mean}, a
        factor of ${row.ratio.mean} with a seed spread of ${row.ratio.spread} under it.
        ${ex ? `Read sideways instead: to reach what ${row.bits} bits of vector quantisation
          reaches, the separable quantiser needs ${ex.scalar_bits_needed} bits, so
          <span class="value">${ex.extra}</span> of them buy nothing but the fact that the two
          coordinates were treated as unrelated.` : ''}
        The last column is the entropy of how often each entry actually gets used: at
        ${row.usage_bits.mean} bits against a nominal ${row.bits}, an entropy coder would send
        this code in fewer bits than the codebook size suggests, and that gap is the one thing
        a fixed length code cannot claim back.
      </div>`;
  }

  slider.addEventListener('input', render);
  render();
  return { render };
}
