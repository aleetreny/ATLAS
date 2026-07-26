/* Four models, four different relationships with the truth, plus the
 * perfectly-calibrated dummy that makes the point that honesty alone is
 * worthless. One reliability curve at a time against the fixed diagonal.
 */
import { drawGrid, drawAxes, axisLabels, makeChart } from '../../assets/js/chart.js';
import { drawDiagonal, drawReliability } from './calkit.js';

const CAST = [
  {
    key: 'logistic', label: 'logistic',
    story: 'The honest one. Fitted on the true 1% prevalence and reading the world through a sigmoid, ' +
      'its claims track reality almost perfectly out of the box. Calibrating it further is polishing a mirror.',
  },
  {
    key: 'naive_bayes', label: 'naive Bayes',
    story: 'The extremist. Its independence assumption multiplies 54 pieces of "evidence" as if they were ' +
      'separate witnesses, when the terrain features echo each other, and every echo compounds the ' +
      'certainty. The result uses only two probabilities: essentially 0, and essentially 1. It stakes ' +
      'total certainty on 7,556 cells, and is wrong about 85% of them.',
  },
  {
    key: 'random_forest', label: 'random forest',
    story: 'The mumbler. Its probability is a vote share across 200 trees, and votes are rarely extreme, ' +
      'so the claims compress toward the middle from both ends: where it says 55%, the truth is 76%; where ' +
      'it says 75%, the truth is 93%. Note its AUROC, though: the forest knows this terrain better than ' +
      'anything else on the page. It just talks about it timidly.',
  },
  {
    key: 'smote_logistic', label: 'SMOTE logistic',
    story: 'The inflated one, carried over from the previous article: trained on a rebalanced world, it ' +
      'overstates everything by roughly the same factor. Of the four, its lie has the simplest shape.',
  },
  {
    key: 'constant', label: 'always say 1%',
    story: 'The warning. It claims 1.00% for every cell, which is exactly the base rate, so it is nearly ' +
      'perfectly calibrated. It is also incapable of telling any two cells apart: AUROC 0.5. Calibration ' +
      'is necessary for trust, and it is not the same thing as knowing anything.',
  },
];

export function initModelsWidget(data) {
  const { g, w, h } = makeChart('#models-chart', {
    width: 620,
    height: 470,
    margin: { top: 30, right: 26, bottom: 56, left: 62 },
  });

  const x = d3.scaleLinear().domain([0, 1]).range([0, w]);
  const y = d3.scaleLinear().domain([0, 1]).range([h, 0]);
  drawGrid(g, x, y, w, h);
  drawAxes(g, x, y, w, h);
  axisLabels(g, w, h, { x: 'probability the model claimed', y: 'share that really was the tree' });
  drawDiagonal(g.append('g'), x, y, w, h);
  const curveG = g.append('g');

  const btnRow = document.querySelector('#models-buttons');
  const readout = document.querySelector('#models-readout');
  const state = { i: 1 };   /* open on naive Bayes, the article's main patient */

  const btns = CAST.map((c, i) => {
    const b = document.createElement('button');
    b.className = 'atlas-btn' + (i === state.i ? '' : ' ghost');
    b.textContent = c.label;
    b.addEventListener('click', () => { state.i = i; render(); });
    btnRow.appendChild(b);
    return b;
  });

  function render() {
    const c = CAST[state.i];
    const m = data.models[c.key].raw;
    drawReliability(curveG, m.bins, x, y, c.key === 'logistic' ? 'var(--primary)' : 'var(--cosmos)');
    btns.forEach((b, i) => b.classList.toggle('ghost', i !== state.i));
    readout.innerHTML =
      `<table class="cal-table">
         <tr><td>ECE</td><td><span class="value">${m.ece.toFixed(4)}</span></td>
             <td>Brier</td><td><span class="value">${m.brier.toFixed(5)}</span></td></tr>
         <tr><td>AUROC</td><td>${m.auroc.toFixed(4)}</td>
             <td>implied base rate</td><td>${(m.mean_p * 100).toFixed(2)}% (truth 1.00%)</td></tr>
       </table>` +
      `<div class="slider-label" style="margin-top:.6rem;line-height:1.5">${c.story}</div>`;
  }

  render();
}
