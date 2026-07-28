/* The floor the alphabet puts under everything above it.
 *
 * An oracle that already knows the next twelve values and only has to write
 * them down in this alphabet still makes an error, and no model reading these
 * letters can do better. The slider is the alphabet size, the error is
 * recomputed here from the same series the generator used, and the readout says
 * how far the real model is from that floor: if the two were close, the answer
 * would be to buy more letters rather than a bigger model.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';
import { encodeWindow, decodeTokens } from './scrolly-token.js';

export function initAlphabetWidget(data) {
  const Q = data.quantisation;
  const D = Q.demo;
  const clip = data.meta.clip;
  const sizes = Q.rows.map((r) => r.bins);
  const slider = document.querySelector('#bins-slider');
  const valueOut = document.querySelector('#bins-value');
  const readout = document.querySelector('#alphabet-readout');
  slider.min = 0;
  slider.max = sizes.length - 1;
  slider.value = String(sizes.indexOf(data.meta.bins));

  const chart = makeChart('#alphabet-chart', {
    width: 640, height: 330, margin: { top: 46, right: 26, bottom: 50, left: 70 },
  });
  const { g, w, h, margin } = chart;
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -26).attr('text-anchor', 'middle').style('font-size', '12px');
  const layer = g.append('g');
  const n = D.window.length;
  const x = d3.scaleLinear().domain([0, n - 1]).range([0, w]);

  function render() {
    const bins = sizes[+slider.value];
    const tokens = encodeWindow(D.window, D.mean, D.sd, bins, clip);
    const back = decodeTokens(tokens, D.mean, D.sd, bins, clip);
    let worst = 0;
    let sum = 0;
    for (let i = 0; i < n; i++) {
      worst = Math.max(worst, Math.abs(back[i] - D.window[i]));
      sum += Math.abs(back[i] - D.window[i]);
    }
    layer.selectAll('*').remove();
    const y = d3.scaleLinear()
      .domain([d3.min(D.window) - 0.6, d3.max(D.window) + 0.6]).range([h, 0]);
    drawGrid(layer, x, y, w, h, { xTicks: 5, yTicks: 5 });
    drawAxes(layer, x, y, w, h, { xTicks: 5, yTicks: 5 });
    axisLabels(layer, w, h, {
      x: 'months of the window', y: 'parts per million', leftBudget: margin.left,
    });
    layer.append('path').datum(D.window).attr('fill', 'none')
      .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.4).attr('opacity', 0.55)
      .attr('d', d3.line().x((d, i) => x(i)).y((d) => y(d)));
    layer.append('path').datum(back).attr('fill', 'none').attr('stroke', 'var(--primary)')
      .attr('stroke-width', 2).attr('d', d3.line().curve(d3.curveStepAfter)
        .x((d, i) => x(i)).y((d) => y(d)));
    layer.append('text').attr('class', 'chart-note').attr('x', 2).attr('y', -8)
      .style('font-size', '11px')
      .text('grey: the real values. accent: what survives the round trip');
    valueOut.textContent = `${bins}`;
    wrapLabel(title, `${bins} letters: the worst value moves ${worst.toFixed(3)} parts `
      + `per million`, w - 8);

    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>letters</th>${Q.rows.map((r) => `<th>${r.bins}</th>`).join('')}</tr>
        <tr><td>step between neighbouring letters, ppm</td>
            ${Q.rows.map((r) => `<td>${r.step_ppm}</td>`).join('')}</tr>
        <tr><td>error of an oracle that knows the future</td>
            ${Q.rows.map((r) => `<td${r.bins === bins ? ' style="font-weight:600"' : ''}>`
    + `${r.mase}</td>`).join('')}</tr>
      </table>
      <div class="gen-note">
        The last row is a lower bound on anything this alphabet can express: it
        is the error made by writing down the <span class="bold">true</span>
        next twelve months in that many letters and reading them back, over
        ${Q.n_origins} origins, in the same units as every other error in this
        branch. At ${bins} letters the round trip on the window drawn above
        costs ${sum.toFixed(3)} parts per million in total and
        ${worst.toFixed(3)} on its worst single month. Drag to the left until
        the accent line stops following the grey one, then read the paragraph
        below for where the model actually sits on this row.
      </div>`;
  }

  slider.addEventListener('input', render);
  render();
  return { render };
}
