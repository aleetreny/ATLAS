/* What the bottleneck costs, three ways, on digits the fit never saw.
 *
 * The three series are PCA, an autoencoder with every activation removed, and
 * the same network with them put back. The first two are on top of each other
 * by a theorem; the gap to the third is the whole answer to "why not just use
 * PCA".
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';

export function initCurveWidget(data) {
  const rows = data.curve.rows;
  const chart = makeChart('#curve-chart', {
    width: 760, height: 380, margin: { top: 76, right: 24, bottom: 58, left: 78 },
  });
  const { g, w, h } = chart;
  const buttons = document.querySelector('#curve-buttons');
  const readout = document.querySelector('#curve-readout');
  const st = { key: 'test' };

  const btns = {};
  for (const [key, text] of [['test', 'on digits it never saw'], ['train', 'on the ones it was fitted to']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn' + (key === st.key ? '' : ' ghost');
    b.textContent = text;
    b.addEventListener('click', () => {
      st.key = key;
      render();
    });
    buttons.appendChild(b);
    btns[key] = b;
  }

  const body = g.append('g');
  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -56).attr('text-anchor', 'middle')
    .style('font-size', '12px').style('text-transform', 'none');
  const legend = g.append('g');

  const SERIES = [
    ['pca', 'PCA', 'var(--anchor)'],
    ['linear', 'an autoencoder with nothing bent', 'var(--smile)'],
    ['deep', 'the same network with the bends put back', 'var(--primary)'],
  ];

  function render() {
    for (const [k, b] of Object.entries(btns)) b.classList.toggle('ghost', k !== st.key);
    body.selectAll('*').remove();
    legend.selectAll('*').remove();
    g.selectAll('g.axis, g.grid, text.axis-label').remove();
    const suffix = st.key === 'test' ? '_test' : '_train';
    const all = rows.flatMap((r) => SERIES.map(([k]) => r[k + suffix]));
    const x = d3.scaleLog().domain([1.7, 38]).range([0, w]);
    const y = d3.scaleLog().domain([d3.min(all) * 0.85, d3.max(all) * 1.15]).range([h, 0]);
    const xs = data.meta.ks;
    const ys = [0.005, 0.01, 0.02, 0.04].filter((v) => v >= d3.min(all) * 0.85
      && v <= d3.max(all) * 1.15);
    drawGrid(g, x, y, w, h, { xValues: xs, yValues: ys });
    drawAxes(g, x, y, w, h, {
      xValues: xs, xFmt: d3.format('d'), yValues: ys, yFmt: d3.format('.3f'),
    });
    axisLabels(g, w, h, { x: 'numbers kept', y: 'squared error per pixel' });
    SERIES.forEach(([k, , c], i) => {
      body.append('path').datum(rows).attr('fill', 'none').attr('stroke', c)
        .attr('stroke-width', 2.6)
        .attr('stroke-dasharray', k === 'linear' ? '7 4' : null)
        .attr('d', d3.line().x((r) => x(r.k)).y((r) => y(r[k + suffix])));
      body.selectAll(`circle.s${i}`).data(rows).join('circle').attr('class', `s${i}`)
        .attr('cx', (r) => x(r.k)).attr('cy', (r) => y(r[k + suffix])).attr('r', 4.5)
        .attr('fill', c).attr('stroke', 'white').attr('stroke-width', 1.3);
      legend.append('line').attr('x1', 2).attr('x2', 24).attr('y1', -34 + i * 15)
        .attr('y2', -34 + i * 15).attr('stroke', c).attr('stroke-width', 2.6)
        .attr('stroke-dasharray', k === 'linear' ? '7 4' : null);
      legend.append('text').attr('x', 30).attr('y', -30 + i * 15)
        .style('font-size', '11.5px').style('font-family', 'var(--font-mono)')
        .attr('fill', 'var(--squidink)').text(SERIES[i][1]);
    });
    wrapLabel(title, st.key === 'test'
      ? `held out digits: ${data.meta.test} of the ${data.meta.n.toLocaleString('en-US')} `
        + 'were kept out of every fit on this chart'
      : `the ${data.meta.train.toLocaleString('en-US')} digits every model on this chart was `
        + 'fitted to', w - 8);

    const best = rows.reduce((a, b) => (b.gain > a.gain ? b : a));
    const two = rows.find((r) => r.k === data.meta.k_show);
    readout.innerHTML = `
      <table class="ae-table">
        <tr><th>numbers kept</th>${rows.map((r) => `<th>${r.k}</th>`).join('')}</tr>
        <tr><td>PCA</td>${rows.map((r) => `<td>${r['pca' + suffix]}</td>`).join('')}</tr>
        <tr><td>nothing bent</td>${rows.map((r) => `<td>${r['linear' + suffix]}</td>`).join('')}</tr>
        <tr><td>bends put back</td>${rows.map((r) => `<td><span class="value">${r['deep' + suffix]}</span></td>`).join('')}</tr>
      </table>
      <div class="ae-note">
        The dashed line sits on top of PCA at every k, which is the theorem: with no activation
        functions the best a network can do is the best a projection can do, and the two agree here
        to within
        ${Math.max(...rows.map((r) => Math.abs(r['linear' + suffix] / r['pca' + suffix] - 1) * 100))
    .toFixed(2)}%. The gap that matters is the third line. Bending the encoder and the decoder buys
        a factor of <span class="bold">${best.gain}</span> at its best, at k = ${best.k}, and
        ${two.gain} at the two numbers the widget above draws. That is real and it is not enormous,
        and the honest way to hold it is that the reason to prefer an autoencoder here is rarely the
        error.
      </div>`;
  }

  render();
}
