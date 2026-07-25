/* What Gaussian naive Bayes actually stores: two bells per feature.
 *
 * The buttons are the features, and the readout says how much separation each
 * one carries on its own, measured rather than asserted, so the reader can see
 * that "mean smoothness" really is nearly useless and "worst perimeter" really
 * is nearly sufficient.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initNbWidget(data) {
  const shapes = data.shapes;
  const { g, w, h } = makeChart('#nb-chart', {
    width: 700,
    height: 380,
    margin: { top: 32, right: 26, bottom: 58, left: 66 },
  });

  const x = d3.scaleLinear().range([0, w]);
  const y = d3.scaleLinear().range([h, 0]);

  const clip = 'nb-clip';
  g.append('clipPath').attr('id', clip).append('rect').attr('x', -1).attr('y', -1)
    .attr('width', w + 2).attr('height', h + 2);
  const plot = g.append('g').attr('clip-path', `url(#${clip})`);

  const area = d3.area().x((d) => x(d[0])).y0(h).y1((d) => y(d[1]));
  const line = d3.line().x((d) => x(d[0])).y((d) => y(d[1]));

  const COLOUR = { benign: 'var(--aqua)', malignant: 'var(--cosmos)' };
  const fills = {};
  const strokes = {};
  for (const cls of ['benign', 'malignant']) {
    fills[cls] = plot.append('path').attr('fill', COLOUR[cls]).attr('opacity', 0.16);
    strokes[cls] = plot.append('path').attr('fill', 'none').attr('stroke', COLOUR[cls]).attr('stroke-width', 3);
  }
  const meanLines = {};
  for (const cls of ['benign', 'malignant']) {
    meanLines[cls] = plot.append('line').attr('y1', 0).attr('y2', h)
      .attr('stroke', COLOUR[cls]).attr('stroke-width', 1.4).attr('stroke-dasharray', '5 4').attr('opacity', 0.75);
  }

  const legend = g.append('g').attr('transform', `translate(${w - 130},4)`);
  ['benign', 'malignant'].forEach((cls, i) => {
    const row = legend.append('g').attr('transform', `translate(0,${i * 18})`);
    row.append('rect').attr('width', 12).attr('height', 12).attr('y', -9).attr('fill', COLOUR[cls]).attr('opacity', 0.45)
      .attr('stroke', COLOUR[cls]).attr('stroke-width', 1.4);
    row.append('text').attr('x', 18).attr('y', 1).attr('class', 'chart-annotation')
      .style('font-size', '0.62rem').style('text-transform', 'none').attr('fill', COLOUR[cls]).text(cls);
  });

  const btnRow = document.querySelector('#nb-buttons');
  const readout = document.querySelector('#nb-readout');
  const buttons = shapes.map((s, i) => {
    const b = document.createElement('button');
    b.className = 'atlas-btn' + (i === 0 ? '' : ' ghost');
    b.textContent = s.feature.replace(/_/g, ' ');
    b.addEventListener('click', () => show(i));
    btnRow.appendChild(b);
    return b;
  });

  /* Overlap between the two bells, the share of one distribution that lands
   * where the other one is denser. 0 means the feature separates the classes
   * perfectly on its own, 1 means it says nothing at all. */
  function overlap(s) {
    const gr = s.grid;
    const a = s.curves.benign.pdf;
    const b = s.curves.malignant.pdf;
    const step = gr[1] - gr[0];
    let inter = 0;
    let total = 0;
    for (let i = 0; i < gr.length; i++) {
      inter += Math.min(a[i], b[i]) * step;
      total += ((a[i] + b[i]) / 2) * step;
    }
    return inter / total;
  }

  function show(i) {
    const s = shapes[i];
    buttons.forEach((b, k) => b.classList.toggle('ghost', k !== i));

    /* A hair of padding, or the leftmost x tick lands on the zero of the y
     * axis where the two meet. */
    const ex = d3.extent(s.grid);
    const pad = (ex[1] - ex[0]) * 0.03;
    x.domain([ex[0] - pad, ex[1] + pad]).nice();

    /* Both bells are divided by the taller one's peak. Densities across these
     * features span four orders of magnitude (area is measured in hundreds,
     * smoothness in hundredths), so the raw axis would print "0.002" on one
     * feature and "40" on the next, and neither number means anything to the
     * reader. Scaling by a single constant leaves the ratio between the two
     * curves untouched, which is the only thing this chart is asking you to
     * read. */
    const peak = d3.max([...s.curves.benign.pdf, ...s.curves.malignant.pdf]);
    y.domain([0, 1.12]);
    drawGrid(g, x, y, w, h, { yTicks: 4 });
    drawAxes(g, x, y, w, h, { yTicks: 4, yFmt: d3.format('.1f') });
    axisLabels(g, w, h, { x: s.feature.replace(/_/g, ' '), y: 'density, tallest peak = 1' });

    for (const cls of ['benign', 'malignant']) {
      const pts = s.grid.map((v, k) => [v, s.curves[cls].pdf[k] / peak]);
      fills[cls].transition('shape').duration(420).attr('d', area(pts));
      strokes[cls].transition('shape').duration(420).attr('d', line(pts));
      meanLines[cls].transition('shape').duration(420)
        .attr('x1', x(s.curves[cls].mu)).attr('x2', x(s.curves[cls].mu));
    }

    const ov = overlap(s);
    const b = s.curves.benign;
    const m = s.curves.malignant;
    const d = Math.abs(m.mu - b.mu) / Math.sqrt((b.sd ** 2 + m.sd ** 2) / 2);
    readout.innerHTML =
      `<table class="pc-table">
         <tr><th></th><th style="color:var(--aqua)">benign</th><th style="color:var(--cosmos)">malignant</th></tr>
         <tr><td>mean</td><td>${b.mu.toFixed(2)}</td><td>${m.mu.toFixed(2)}</td></tr>
         <tr><td>std deviation</td><td>${b.sd.toFixed(2)}</td><td>${m.sd.toFixed(2)}</td></tr>
       </table>` +
      `<div class="slider-label" style="margin-top:.6rem;line-height:1.45">` +
      `Four numbers, and that is the entire model for this feature. The two bells overlap over ` +
      `<span class="value">${(ov * 100).toFixed(0)}%</span> of their mass ` +
      `(a separation of <span class="value">${d.toFixed(2)}</span> standard deviations), so on its own this ` +
      `feature is ${ov < 0.35 ? 'close to sufficient' : ov < 0.6 ? 'informative but not decisive' : 'nearly silent'}. ` +
      `Naive Bayes multiplies thirty of these together and assumes none of them are saying the same thing twice.</div>`;
  }

  show(0);
}
