/* Entry point.
 *
 * Two of the guards here check arithmetic the page argues from and could not
 * otherwise notice: the bits per weight, which is the only claim on the page
 * with an exact answer, and the ordering of the truncation ceiling, which has
 * to be monotone because a rank r+1 approximation contains the rank r one.
 */
import { initSpecWidget } from './spec-widget.js';
import { initRankWidget } from './rank-widget.js';
import { initAlphaWidget } from './alpha-widget.js';
import { initQuantWidget } from './quant-widget.js';
import { initMemWidget } from './mem-widget.js';
import { initProse, PROSE_IDS } from './prose.js';

function renderMath() {
  if (typeof renderMathInElement !== 'function') return;
  renderMathInElement(document.body, {
    delimiters: [
      { left: '$$', right: '$$', display: true },
      { left: '\\(', right: '\\)', display: false },
    ],
    throwOnError: false,
  });
}

function safely(name, fn) {
  try {
    return fn();
  } catch (err) {
    console.error(`widget "${name}" failed to initialise:`, err);
    return null;
  }
}

function check(label, ok, detail) {
  if (!ok) console.warn(`[lora] ${label}: ${detail}`);
  return ok;
}

function runGuards(data) {
  data.spectra.layers.forEach((L) => {
    ['delta', 'base', 'random'].forEach((k) => {
      const e = L[k].energy;
      const rising = e.every((v, i) => i === 0 || v >= e[i - 1] - 1e-9);
      check(`layer ${L.layer}: the ${k} energy curve only goes up`, rising,
        'a cumulative share of a total went down, so the singular values are not sorted');
      /* the curve is exported truncated, so it can only confirm an r90 that falls
         inside it; past the end all it can say is that the file agrees the
         answer is further out than the page can draw */
      const hit = e.findIndex((v) => v >= 0.9);
      check(`layer ${L.layer}: directions for ninety per cent of the ${k}`,
        hit >= 0 ? hit + 1 === L[k].r90 : L[k].r90 > e.length,
        hit >= 0 ? `the page counts ${hit + 1} and the file says ${L[k].r90}`
          : `the drawn curve never reaches ninety per cent in ${e.length} directions, `
            + `so the file's ${L[k].r90} has to be past ${e.length} and is not`);
    });
    /* the null has to be a null: a random matrix cannot be MORE concentrated
       than the pretrained weights it is standing in for, or it is not a null */
    check(`layer ${L.layer}: the random control is flat`,
      L.random.eff >= L.base.eff * 0.5,
      `its effective rank is ${L.random.eff} against ${L.base.eff} for the real matrix`);
  });

  /* truncation is nested, so the energy kept can only rise with the rank */
  const c = data.ceiling;
  const bad = c.filter((r, i) => i > 0 && r.kept < c[i - 1].kept - 1e-9).length;
  check('the truncation ceiling keeps more energy at every rank', bad === 0,
    `${bad} of ${c.length} ranks keep less energy than the rank below them`);

  /* bits per weight, which is the one number on the page with an exact answer */
  data.quant.blocks.forEach((b) => {
    const plain = 4 + 32 / b.block;
    check(`bits per weight at block ${b.block}`, Math.abs(plain - b.bits) < 1e-6,
      `four bits plus one 32 bit scale per ${b.block} weights is ${plain.toFixed(4)}, `
      + `and the file says ${b.bits}`);
    check(`double quantisation saves at block ${b.block}`, b.bits_double < b.bits,
      `${b.bits_double} is not below ${b.bits}`);
  });

  /* the level counts the readout quotes */
  check('the quantile format has sixteen levels', data.quant.nf4.length === 16,
    `${data.quant.nf4.length} levels`);
  check('the floating point format wastes a code on negative zero',
    data.quant.fp4.length === 15, `${data.quant.fp4.length} levels rather than 15`);

  /* the memory table, which is arithmetic and therefore exactly checkable */
  data.memory.rows.forEach((row) => {
    check(`the memory total for ${row.arm}`,
      row.total === row.weights + row.grads + row.optimiser,
      `${row.total} against ${row.weights + row.grads + row.optimiser}`);
    check(`the optimiser state for ${row.arm}`, row.optimiser === 2 * row.grads,
      'Adam keeps two moments per trainable parameter, so it must be twice the gradients');
  });

  const badText = [];
  document.querySelectorAll('p.body-text, .widget-readout, .atlas-table td, .gen-note')
    .forEach((node) => {
      const t = node.textContent || '';
      if (/\bundefined\b|\bNaN\b|\[object Object\]/.test(t)) {
        badText.push(node.id || node.className);
      }
    });
  check('composed text', badText.length === 0,
    `undefined or NaN appears in: ${badText.join(', ')}`);

  console.info('[lora] load-time checks complete');
}

async function boot() {
  renderMath();
  const data = await fetch('./data/lora.json').then((r) => r.json());

  safely('prose', () => initProse(data));
  safely('the spectra', () => initSpecWidget(data));
  safely('rank against budget', () => initRankWidget(data));
  safely('alpha', () => initAlphaWidget(data));
  safely('four bits', () => initQuantWidget(data));
  safely('the memory', () => initMemWidget(data));

  renderMath();

  window.__atlasProseIds = PROSE_IDS;
  window.__atlasCheck = () => safely('guards', () => runGuards(data));
  const guards = () => window.__atlasCheck();
  if ('requestIdleCallback' in window) requestIdleCallback(guards, { timeout: 5000 });
  else setTimeout(guards, 1000);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
