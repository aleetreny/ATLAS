/* Prediction sets on real digits, recomputed as the reader moves the level.
 *
 * The digits are drawn because a set of labels is abstract and an ambiguous
 * eight is not: the widget's job is to make "the set grew" mean "this one is
 * genuinely hard to tell from a three".
 */
import { makeChart } from '../../assets/js/chart.js';
import { quantileAt } from './conformalkit.js';

const f2 = (v) => v.toFixed(2);
const pc = (v, d = 0) => `${(100 * v).toFixed(d)}%`;
const LEVELS = [0.2, 0.15, 0.1, 0.05, 0.01];

export function initSetsWidget(data) {
  const host = document.querySelector('#sets-widget');
  if (!host) return;
  const S = data.classification;
  const D = S.demo;
  const chart = makeChart('#sets-chart', {
    /* La altura sale de lo que ocupa una fila y no al reves: ocho pixeles de
     * once unidades, ocho de hueco y quince de etiqueta son 111 por fila, y
     * con cuatro filas la caja tiene que medir 444 mas los margenes. Con 380
     * la ultima fila de etiquetas caia fuera del viewBox. */
    width: 640, height: 500, margin: { top: 26, right: 20, bottom: 20, left: 20 },
  });
  const { g, w, h } = chart;
  const readout = document.querySelector('#sets-readout');
  const label = document.querySelector('#sets-alpha-label');
  const slider = document.querySelector('#sets-alpha');
  let i = 2;

  const sorted = [...D.cal_scores].sort((a, b) => a - b);
  const COLS = 5;
  const cell = w / COLS;

  function render() {
    const alpha = LEVELS[i];
    label.textContent = pc(1 - alpha);
    g.selectAll('*').remove();
    const q = quantileAt(sorted, alpha);
    let total = 0;
    let covered = 0;

    D.p.forEach((probs, k) => {
      const row = Math.floor(k / COLS);
      const col = k % COLS;
      const x0 = col * cell;
      const y0 = row * (h / 4);
      const set = probs.map((p, c) => [c, p]).filter(([, p]) => 1 - p <= q).map(([c]) => c);
      total += set.length;
      if (set.includes(D.y[k])) covered += 1;

      /* The digit itself, eight by eight, as one path per grey level: the
       * alternative is 64 rects per digit and twenty digits, which is a
       * kilobyte of DOM for a thumbnail. */
      const img = D.images[k];
      const px = 11;
      const ox = x0 + (cell - 8 * px) / 2;
      const oy = y0 + 8;
      for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
          const v = img[r * 8 + c] / 16;
          if (v < 0.06) continue;
          g.append('rect')
            .attr('x', ox + c * px).attr('y', oy + r * px)
            .attr('width', px).attr('height', px)
            .attr('fill', 'var(--squidink)').attr('fill-opacity', v);
        }
      }
      g.append('text').attr('class', 'chart-note')
        .attr('x', x0 + cell / 2).attr('y', oy + 8 * px + 15)
        .attr('text-anchor', 'middle').style('font-size', '12px')
        .attr('fill', set.includes(D.y[k]) ? 'var(--primary)' : 'var(--cosmos)')
        .text(set.length ? `{${set.join(', ')}}` : 'empty');
    });

    const size = total / D.p.length;
    readout.innerHTML =
      `<p>Twenty held out digits, with the label set the method returns at `
      + `<span class="bold">${pc(1 - alpha)}</span> confidence printed underneath each one. `
      + `The rule is the same as for the interval: keep every label whose score falls under `
      + `the <span class="bold">${f2(q)}</span> cut, which is the `
      + `${quantileAt(sorted, alpha) === Infinity ? 'unreachable' : ''} quantile of the `
      + `calibration scores and is recomputed in this page every time you move the slider. `
      + `Across these twenty the average set holds `
      + `<span class="bold">${f2(size)}</span> labels and the true one is inside `
      + `${covered} times. Push the confidence up and the sets swell around the ambiguous `
      + `digits and stay singletons around the clean ones, which is the useful behaviour: `
      + `the size of the set is the model saying how sure it is, in units anybody can act `
      + `on.</p>`;
  }

  slider.addEventListener('input', (e) => { i = +e.target.value; render(); });
  render();
}
