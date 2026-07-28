/* The same vowel at two pitches, and the same pitch at two vowels.
 *
 * Four sounds, two things that change, and three views of each: the spectrum,
 * the log mel energies, and the cepstral coefficients. The point is which of
 * the three moves when the pitch changes and which moves when the vowel does,
 * and it is visible without any statistics at all.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';
import { decodeInt16, play } from '../../assets/js/dsp.js';

const KEYS = [
  { key: 'a100', label: '"a" at 100 Hz', vowel: 'a', f0: 100 },
  { key: 'a200', label: '"a" at 200 Hz', vowel: 'a', f0: 200 },
  { key: 'i100', label: '"i" at 100 Hz', vowel: 'i', f0: 100 },
  { key: 'i200', label: '"i" at 200 Hz', vowel: 'i', f0: 200 },
];
const VIEWS = [['the spectrum', 'spec'], ['log mel energies', 'mel'], ['the coefficients', 'coef']];

export function initMfccWidget(data) {
  const C = data.cepstrum;
  let current = 0;
  let view = 'spec';
  const controls = d3.select('#mfcc-controls');
  const readout = document.querySelector('#mfcc-readout');
  const chart = makeChart('#mfcc-chart', {
    width: 640, height: 340, margin: { top: 46, right: 26, bottom: 50, left: 70 },
  });
  const { g, w, h, margin } = chart;
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -26).attr('text-anchor', 'middle').style('font-size', '12px');
  const layer = g.append('g');

  const dist = (a, b) => {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
    return s / a.length;
  };

  function render() {
    const K = KEYS[current];
    const D = C.demo[K.key];
    layer.selectAll('*').remove();
    if (view === 'spec') {
      const x = d3.scaleLinear().domain([0, data.meta.sr / 2]).range([0, w]);
      const y = d3.scaleLinear()
        .domain([d3.min(D.db) - 2, d3.max(D.db) + 4]).range([h, 0]);
      drawGrid(layer, x, y, w, h, { xTicks: 5, yTicks: 4 });
      drawAxes(layer, x, y, w, h, { xTicks: 5, yTicks: 4 });
      axisLabels(layer, w, h, {
        x: 'frequency, hertz', y: 'decibels', leftBudget: margin.left,
      });
      layer.append('path').datum(D.db).attr('fill', 'none').attr('stroke', 'var(--primary)')
        .attr('stroke-width', 1.3)
        .attr('d', d3.line().x((d, i) => x(C.freqs[i])).y((d) => y(d)));
      const env = C.envelope[K.vowel];
      const shift = d3.max(D.db) - d3.max(env);
      layer.append('path').datum(env.map((v) => v + shift)).attr('fill', 'none')
        .attr('stroke', 'var(--anchor)').attr('stroke-width', 2).attr('stroke-dasharray', '6 4')
        .attr('d', d3.line().x((d, i) => x(C.freqs[i])).y((d) => y(d)));
      data.meta.vowels[K.vowel].slice(0, 2).forEach((f, i) => {
        layer.append('text').attr('class', 'chart-note').attr('x', x(f)).attr('y', 12)
          .attr('text-anchor', 'middle').style('font-size', '10.5px')
          .attr('fill', 'var(--anchor)').text(`F${i + 1}`);
      });
      layer.append('text').attr('class', 'chart-note').attr('x', 2).attr('y', -8)
        .style('font-size', '11px').attr('fill', 'var(--anchor)')
        .text('dashed: the filter this sound was actually made with');
    } else if (view === 'mel') {
      const x = d3.scaleLinear().domain([0.5, D.logmel.length + 0.5]).range([0, w]);
      const all = KEYS.flatMap((k) => C.demo[k.key].logmel);
      const y = d3.scaleLinear().domain([d3.min(all) - 1, d3.max(all) + 1]).range([h, 0]);
      drawGrid(layer, x, y, w, h, { xTicks: 6, yTicks: 4 });
      drawAxes(layer, x, y, w, h, { xTicks: 6, yTicks: 4 });
      axisLabels(layer, w, h, {
        x: 'mel band', y: 'log energy', leftBudget: margin.left,
      });
      layer.selectAll('rect').data(D.logmel).join('rect')
        .attr('x', (d, i) => x(i + 1) - w / (2 * D.logmel.length) + 1)
        .attr('width', w / D.logmel.length - 2)
        .attr('y', (d) => Math.min(y(d), y(y.domain()[0])))
        .attr('height', (d) => Math.abs(y(d) - y(y.domain()[0])))
        .attr('fill', 'var(--primary)').attr('opacity', 0.85);
    } else {
      const x = d3.scaleLinear().domain([-0.5, D.coef.length - 0.5]).range([0, w]);
      const all = KEYS.flatMap((k) => C.demo[k.key].coef.slice(1));
      const y = d3.scaleLinear().domain([d3.min(all) * 1.2, d3.max(all) * 1.2]).range([h, 0]);
      drawGrid(layer, x, y, w, h, { xTicks: 6, yTicks: 4 });
      drawAxes(layer, x, y, w, h, { xTicks: 6, yTicks: 4 });
      axisLabels(layer, w, h, {
        x: 'coefficient', y: 'value', leftBudget: margin.left,
      });
      layer.append('line').attr('x1', 0).attr('x2', w).attr('y1', y(0)).attr('y2', y(0))
        .attr('stroke', 'var(--squidink)').attr('opacity', 0.5);
      layer.selectAll('rect').data(D.coef.slice(1)).join('rect')
        .attr('x', (d, i) => x(i + 1) - 10).attr('width', 20)
        .attr('y', (d) => Math.min(y(d), y(0))).attr('height', (d) => Math.abs(y(d) - y(0)))
        .attr('fill', 'var(--primary)');
      layer.append('text').attr('class', 'chart-note').attr('x', 2).attr('y', -8)
        .style('font-size', '11px')
        .text('the zeroth coefficient is loudness and is not drawn');
    }
    wrapLabel(title, `${K.label}, as ${VIEWS.find((v) => v[1] === view)[0]}`, w - 8);

    const same = C.demo[K.vowel === 'a' ? (K.f0 === 100 ? 'a200' : 'a100')
      : (K.f0 === 100 ? 'i200' : 'i100')];
    const other = C.demo[K.vowel === 'a' ? `i${K.f0}` : `a${K.f0}`];
    readout.innerHTML = `
      <table class="gen-table">
        <tr><th>distance to</th><th>the spectrum</th><th>the coefficients</th></tr>
        <tr><td>the same vowel at the other pitch</td>
            <td><span class="value">${dist(D.db, same.db).toFixed(3)}</span></td>
            <td><span class="value">${dist(D.coef.slice(1), same.coef.slice(1)).toFixed(3)}</span></td></tr>
        <tr><td>the other vowel at this pitch</td>
            <td><span class="value">${dist(D.db, other.db).toFixed(3)}</span></td>
            <td><span class="value">${dist(D.coef.slice(1), other.coef.slice(1)).toFixed(3)}</span></td></tr>
      </table>
      <div class="gen-note">
        Those four numbers are the whole argument for the cepstrum, and they are
        recomputed here for whichever sound is selected. What you want is the
        second row much larger than the first, because a vowel should look like
        itself at any pitch, and you want that ratio to be better in the right
        hand column than in the left. Change the vowel and the pitch and watch
        whether it stays true; the paragraph below has the same comparison run
        over all ${C.pitches.length} pitches and all five vowels at once.
      </div>`;
  }

  controls.selectAll('button.snd').data(KEYS).join('button')
    .attr('class', (d, i) => `atlas-btn snd${i === current ? '' : ' ghost'}`)
    .text((d) => d.label)
    .on('click', (evt, d) => {
      current = KEYS.indexOf(d);
      controls.selectAll('button.snd')
        .attr('class', (dd) => `atlas-btn snd${dd === d ? '' : ' ghost'}`);
      render();
    });
  controls.selectAll('button.view').data(VIEWS).join('button')
    .attr('class', (d) => `atlas-btn view${d[1] === view ? '' : ' ghost'}`)
    .text((d) => d[0])
    .on('click', (evt, d) => {
      view = d[1];
      controls.selectAll('button.view')
        .attr('class', (dd) => `atlas-btn view${dd[1] === view ? '' : ' ghost'}`);
      render();
    });
  controls.append('button').attr('class', 'atlas-btn ghost').text('listen')
    .on('click', () => play(decodeInt16(C.demo[KEYS[current].key].audio), data.meta.sr));

  render();
  return { render };
}
