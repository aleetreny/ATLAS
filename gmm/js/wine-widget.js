/* The wine, softly: the blind test of the last article rerun with a model
 * that is allowed to hesitate.
 *
 * Three views. Certainty: every bottle tinted by its full membership vector,
 * ring size marking the hesitant ones. Hesitations: the bottles under 90%
 * membership, with the measured payoff of listening to them. Mistakes: the
 * five errors, including the humbling one made with total confidence.
 */
import { drawGrid, drawAxes, axisLabels, makeChart } from '../../assets/js/chart.js';
import { COLOURS, softColour } from './mixdraw.js';

export function initWineWidget(data) {
  const W = data.wine;
  const { g, w, h } = makeChart('#wine-chart', {
    width: 660,
    height: 460,
    margin: { top: 34, right: 26, bottom: 56, left: 58 },
  });

  const x = d3.scaleLinear().domain(d3.extent(W.proj, (p) => p[0])).nice().range([0, w]);
  const y = d3.scaleLinear().domain(d3.extent(W.proj, (p) => p[1])).nice().range([h, 0]);
  drawGrid(g, x, y, w, h);
  drawAxes(g, x, y, w, h);
  axisLabels(g, w, h, { x: 'first principal component', y: 'second principal component' });

  const dotG = g.append('g');
  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -14).attr('text-anchor', 'middle').style('font-size', '0.66rem');

  const readout = document.querySelector('#winegmm-readout');
  const btns = {
    soft: document.querySelector('#winegmm-soft'),
    unsure: document.querySelector('#winegmm-unsure'),
    wrong: document.querySelector('#winegmm-wrong'),
  };
  const state = { view: 'soft' };

  const top = W.probs.map((p) => Math.max(...p));
  const unsure = W.proj.map((_, i) => top[i] < 0.9);
  const wrong = W.proj.map((_, i) => W.gmm[i] !== W.true[i]);
  const nUnsure = unsure.filter(Boolean).length;
  const rightSure = W.proj.filter((_, i) => !unsure[i] && !wrong[i]).length;
  const nSure = W.proj.filter((_, i) => !unsure[i]).length;
  const rightUnsure = W.proj.filter((_, i) => unsure[i] && !wrong[i]).length;
  const confidentWrong = W.proj.map((_, i) => wrong[i] && top[i] >= 0.99).filter(Boolean).length;

  function render() {
    const v = state.view;
    dotG.selectAll('circle').data(W.proj).join('circle')
      .attr('cx', (p) => x(p[0]))
      .attr('cy', (p) => y(p[1]))
      .attr('r', (p, i) =>
        v === 'unsure' ? (unsure[i] ? 7 : 3.4)
          : v === 'wrong' ? (wrong[i] ? 7 : 3.4)
            : 3 + 5 * (1 - top[i]))
      .attr('fill', (p, i) => (v === 'wrong' && !wrong[i] ? 'var(--stone)' : softColour(W.probs[i])))
      .attr('stroke', (p, i) =>
        (v === 'unsure' && unsure[i]) || (v === 'wrong' && wrong[i]) ? 'var(--squidink)' : 'white')
      .attr('stroke-width', (p, i) =>
        (v === 'unsure' && unsure[i]) || (v === 'wrong' && wrong[i]) ? 1.8 : 0.6)
      .attr('opacity', (p, i) => (v === 'unsure' && !unsure[i] ? 0.35 : 0.88));

    for (const [k, b] of Object.entries(btns)) b.classList.toggle('ghost', k !== v);
    title.text(
      v === 'soft' ? 'every bottle tinted by its membership vector; bigger = less sure'
        : v === 'unsure' ? `the ${nUnsure} bottles below 90% membership`
          : `the ${wrong.filter(Boolean).length} disagreements with the cultivar labels`
    );

    readout.innerHTML =
      `<table class="mix-table">
         <tr><td>agreement</td><td><span class="value">${W.agree}</span> of 178 (k-means: 172)</td>
             <td>adjusted Rand index</td><td><span class="value">${W.ari.toFixed(3)}</span> (k-means: ${W.km_ari.toFixed(3)})</td></tr>
       </table>` +
      `<div class="slider-label" style="margin-top:.6rem;line-height:1.5">${
        v === 'soft'
          ? `Clustered blind on all 13 standardised dimensions with the covariance family BIC chose ` +
            `(${W.chosen_type}: the full family's 314 parameters overfit 178 bottles and score ` +
            `ARI ${W.compare_ari.full.toFixed(3)}). The picture only looks flat-coloured because the ` +
            `mixture is genuinely sure about most bottles: the mean top membership is 0.991.`
          : v === 'unsure'
            ? `Here is the payoff of a model that hesitates: among the ${nSure} bottles it is sure about, ` +
              `it is right on ${rightSure} (${(rightSure / nSure * 100).toFixed(1)}%). Among these ` +
              `${nUnsure}, only ${rightUnsure} of ${nUnsure}. The hesitation is information: ship these ` +
              `bottles to a human, trust the rest.`
            : `Five errors to k-means' six, and four of k-means' mistakes fixed. The humbling part: ` +
              `${confidentWrong} of the five are made with essentially total confidence, so soft ` +
              `assignment narrows the risk, it does not abolish it. Membership probabilities are model ` +
              `beliefs, and the calibration article said everything that needs saying about beliefs.`
      }</div>`;
  }

  for (const [k, b] of Object.entries(btns)) {
    b.addEventListener('click', () => { state.view = k; render(); });
  }
  render();
}
