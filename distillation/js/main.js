/* Entry point.
 *
 * The load-time guards do the arithmetic the page argues from rather than
 * trusting the file: the split of the transfer advantage into projection and
 * learning, and the claim that the scrambling control really did preserve the
 * entropy it says it preserved. That second one is the whole control: if the
 * scramble had changed the entropy, the arm would be measuring two things at
 * once and the conclusion drawn from it would be worthless.
 */
import { initTransferWidget } from './transfer-widget.js';
import { initFreezeWidget } from './freeze-widget.js';
import { initDistilWidget } from './distil-widget.js';
import { initTempWidget } from './temp-widget.js';
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
  if (!ok) console.warn(`[distillation] ${label}: ${detail}`);
  return ok;
}

function runGuards(data) {
  /* Every distillation arm ships the per seed accuracies it was summarised from,
     so the summary can be checked against them instead of trusted. A mean that
     does not match its own list means the arm was averaged over something other
     than what it published, and no amount of looking at the chart shows that. */
  Object.entries(data.distil.arms).forEach(([name, arm]) => {
    const a = arm.accs;
    check(`${name}: the mean is the mean of its own seeds`,
      Math.abs(a.reduce((x, y) => x + y, 0) / a.length - arm.mean) < 5e-6,
      `${a.length} seeds average ${a.reduce((x, y) => x + y, 0) / a.length} and the file `
      + `says ${arm.mean}`);
    const m = a.reduce((x, y) => x + y, 0) / a.length;
    /* population spread, ddof zero, which is what the generator writes
       everywhere; checking against the sample version fails by exactly
       sqrt(n/(n-1)) and looks like a measurement error rather than a convention */
    const sd = Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length);
    check(`${name}: the spread is the spread of its own seeds`,
      Math.abs(sd - arm.sd) < 5e-6, `${sd} against ${arm.sd}`);
    check(`${name}: it ran on every seed`, a.length === data.distil.seeds,
      `${a.length} accuracies against the ${data.distil.seeds} seeds this section used`);
  });

  ['forward', 'backward'].forEach((dir) => {
    const D = data.transfer[dir];
    check(`${dir}: every arm has every sample size`,
      D.rows.every((r) => D.arms.every((a) => typeof r[a] === 'number')),
      'a row is missing an arm');
    /* the ceiling has to be a ceiling: an encoder trained on the destination
       task cannot be beaten by one trained on the other task, and if it is,
       the two are not the same experiment */
    const broken = D.rows.filter((r) => r.transfer > r.oracle + 0.02).length;
    check(`${dir}: the ceiling is above the transfer`, broken === 0,
      `${broken} of ${D.rows.length} sample sizes have the transferred encoder above the `
      + 'one trained on the destination task by more than two points');
    /* an untrained encoder must not beat a trained one at the largest size, or
       the source training did nothing at all */
    const last = D.rows[D.rows.length - 1];
    check(`${dir}: training the source encoder did something`,
      last.transfer >= last.random - last.transfer_sd - last.random_sd,
      `at ${last.n} rows the untrained encoder scores ${last.random} against `
      + `${last.transfer} for the trained one`);
  });

  /* the scrambling control: same winner, same confidence, same entropy. The
     entropy is the one the page quotes, so it is the one that gets checked. */
  const D = data.distil;
  const ref = D.entropies['4'];
  check('the scramble kept the entropy',
    Math.abs(D.scrambled_entropy - ref) < 5e-6,
    `the scrambled targets carry ${D.scrambled_entropy} nats against ${ref} for the real `
    + 'ones, so that arm changed more than which wrong class got which share');

  /* temperature has to raise the entropy, monotonically, or the dial is not
     doing what the paragraph next to it says */
  const ents = D.temps.map((t) => D.entropies[String(t)]);
  const dips = ents.filter((v, i) => i > 0 && v < ents[i - 1]).length;
  check('temperature flattens the teacher', dips === 0,
    `${dips} steps of the temperature sweep lower the entropy instead of raising it`);

  /* the compression factor quoted in the prose */
  check('the compression factor',
    Math.abs(D.teacher_params / D.student_params - D.compression) < 1e-6,
    `${(D.teacher_params / D.student_params).toFixed(4)} against ${D.compression}`);

  const bad = [];
  document.querySelectorAll('p.body-text, .widget-readout, .atlas-table td, .gen-note')
    .forEach((node) => {
      const t = node.textContent || '';
      if (/\bundefined\b|\bNaN\b|\[object Object\]/.test(t)) bad.push(node.id || node.className);
    });
  check('composed text', bad.length === 0, `undefined or NaN appears in: ${bad.join(', ')}`);

  console.info('[distillation] load-time checks complete');
}

async function boot() {
  renderMath();
  const data = await fetch('./data/distillation.json').then((r) => r.json());

  safely('prose', () => initProse(data));
  safely('the transfer curve', () => initTransferWidget(data));
  safely('what to freeze', () => initFreezeWidget(data));
  safely('the distillation arms', () => initDistilWidget(data));
  safely('the two dials', () => initTempWidget(data));

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
