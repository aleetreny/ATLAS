/* Before and after: pick a patient, pick a treatment, read the bill.
 *
 * The raw curve stays as a faint reference whenever a repair is shown, so the
 * eye measures the correction, not just the result. For the SMOTE model a
 * third "logit shift" treatment appears: the analytic fix that uses no
 * calibration data at all, just the arithmetic of the known resampling ratio.
 */
import { drawGrid, drawAxes, axisLabels, makeChart } from '../../assets/js/chart.js';
import { drawDiagonal, drawReliability } from './calkit.js';

const PATIENTS = [
  { key: 'naive_bayes', label: 'naive Bayes' },
  { key: 'random_forest', label: 'random forest' },
  { key: 'smote_logistic', label: 'SMOTE logistic' },
  { key: 'logistic', label: 'logistic' },
];
const TREATMENTS = [
  { key: 'raw', label: 'untreated' },
  { key: 'platt', label: 'Platt' },
  { key: 'isotonic', label: 'isotonic' },
  { key: 'logit_shift', label: 'logit shift', only: 'smote_logistic' },
];

export function initRepairWidget(data) {
  const { g, w, h } = makeChart('#repair-chart', {
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
  const rawG = g.append('g');
  const curveG = g.append('g');

  const pRow = document.querySelector('#repair-patients');
  const tRow = document.querySelector('#repair-treatments');
  const readout = document.querySelector('#repair-readout');
  const state = { p: 'naive_bayes', t: 'raw' };

  const pBtns = PATIENTS.map((p) => {
    const b = document.createElement('button');
    b.className = 'atlas-btn' + (p.key === state.p ? '' : ' ghost');
    b.textContent = p.label;
    b.addEventListener('click', () => { state.p = p.key; render(); });
    pRow.appendChild(b);
    return b;
  });
  const tBtns = TREATMENTS.map((t) => {
    const b = document.createElement('button');
    b.className = 'atlas-btn' + (t.key === state.t ? '' : ' ghost');
    b.textContent = t.label;
    b.addEventListener('click', () => { state.t = t.key; render(); });
    tRow.appendChild(b);
    return b;
  });

  function verdict(p, t, m, raw) {
    if (t === 'raw') {
      return 'The patient before treatment. Both repairs below were fitted on the 41,205 held-out ' +
        'calibration cells, never on the test set the curve is measured on.';
    }
    if (t === 'logit_shift') {
      return `No calibration data used: since SMOTE trained the model at 50% prevalence in a 1% world, ` +
        `subtracting the log odds of that ratio from every logit is the exact algebraic undo. ECE ` +
        `${raw.ece.toFixed(4)} to ${m.ece.toFixed(4)}, from arithmetic alone. When you know how the ` +
        `prior was distorted, you do not need to learn the fix.`;
    }
    const eceGain = raw.ece / Math.max(m.ece, 1e-9);
    const aurocNote = Math.abs(m.auroc - raw.auroc) < 0.0005
      ? 'AUROC is untouched: both repairs are monotone, so the ranking of cells survives.'
      : `AUROC moved from ${raw.auroc.toFixed(4)} to ${m.auroc.toFixed(4)}: isotonic's flat treads tie ` +
        `cells together, and ties cost a little ranking resolution.`;
    return `${t === 'platt' ? 'A two-parameter sigmoid on the logits' : 'The monotone staircase from the widget above'}, ` +
      `fitted on the calibration split: ECE ${raw.ece.toFixed(4)} to ${m.ece.toFixed(4)}` +
      `${eceGain > 3 ? `, ${eceGain.toFixed(0)} times smaller` : ''}. ${aurocNote}`;
  }

  function render() {
    const treats = TREATMENTS.filter((t) => !t.only || t.only === state.p);
    if (!treats.some((t) => t.key === state.t)) state.t = 'raw';
    const m = data.models[state.p][state.t];
    const raw = data.models[state.p].raw;

    if (state.t !== 'raw') {
      drawReliability(rawG, raw.bins, x, y, 'var(--cosmos)', { opacity: 0.25, dots: false, width: 2 });
    } else {
      rawG.selectAll('*').remove();
    }
    drawReliability(curveG, m.bins, x, y, state.t === 'raw' ? 'var(--cosmos)' : 'var(--primary)');

    pBtns.forEach((b, i) => b.classList.toggle('ghost', PATIENTS[i].key !== state.p));
    tBtns.forEach((b, i) => {
      const t = TREATMENTS[i];
      b.style.display = !t.only || t.only === state.p ? '' : 'none';
      b.classList.toggle('ghost', t.key !== state.t);
    });

    readout.innerHTML =
      `<table class="cal-table">
         <tr><td>ECE</td><td><span class="value">${m.ece.toFixed(4)}</span></td>
             <td>Brier</td><td><span class="value">${m.brier.toFixed(5)}</span></td></tr>
         <tr><td>AUROC</td><td>${m.auroc.toFixed(4)}</td>
             <td>implied base rate</td><td>${(m.mean_p * 100).toFixed(2)}% (truth 1.00%)</td></tr>
       </table>` +
      `<div class="slider-label" style="margin-top:.6rem;line-height:1.5">${verdict(state.p, state.t, m, raw)}</div>`;
  }

  render();
}
