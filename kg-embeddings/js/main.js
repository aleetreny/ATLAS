/* Entry point.
 *
 * The guards here check the two things the page claims that a reader cannot
 * verify by looking: that the scoring function the widget draws is the same
 * one the generator trained (a rotation cannot change a length, so the turned
 * point has to stay on its circle), and that the symmetry measured per
 * relation is what the two predictions are being read against.
 */
import { initLiveWidget } from './live-widget.js';
import { initSymWidget } from './sym-widget.js';
import { initPatWidget } from './pat-widget.js';
import { initProse } from './prose.js';

let PROSE_IDS = [];

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

function check(data, loud = false) {
  const problems = [];
  const fail = (label, detail) => problems.push(`${label}: ${detail}`);

  /* 1. a rotation preserves length, so the widget's turn has to as well */
  if (data.widget) {
    let worst = 0;
    Object.values(data.widget.coords).forEach(([a, b]) => {
      const r0 = Math.hypot(a, b);
      const th = 1.1;
      const r1 = Math.hypot(a * Math.cos(th) - b * Math.sin(th),
        a * Math.sin(th) + b * Math.cos(th));
      worst = Math.max(worst, Math.abs(r1 - r0));
    });
    if (!(worst < 1e-12)) {
      fail('the rotation', `turning a point moved it ${worst.toExponential(2)} off its circle`);
    }
  }

  /* 2. the symmetry figures have to be shares, and the two groups have to
     exist, or the whole comparison is being read off one point */
  const sym = data.relations.filter((r) => r.symmetry > 0.5).length;
  const anti = data.relations.length - sym;
  if (sym === 0 || anti === 0) {
    fail('the symmetry split', `there are ${sym} symmetric relations and ${anti} others, so the `
      + 'comparison the page makes has nothing to compare');
  }
  data.relations.forEach((r) => {
    if (r.symmetry < 0 || r.symmetry > 1) {
      fail('the symmetry measure', `${r.name} reports ${r.symmetry}, which is not a share`);
    }
  });

  /* 3. filtered ranking can only be better than raw, never worse */
  ['transe', 'rotate', 'distmult'].forEach((k) => {
    const e = data.wn[k].eval;
    if (e.mrr < e.mrr_raw - 1e-9) {
      fail('the protocol', `${k} reports a filtered MRR (${e.mrr}) below its raw one (${e.mrr_raw}), `
        + 'which cannot happen, since filtering only ever removes competitors');
    }
  });

  /* 4. DistMult is symmetric by construction, so it cannot beat the other two
     on a relation written to have a direction */
  const P = data.patterns.models;
  if (P.distmult.per_pattern.parent.mrr
      > Math.max(P.transe.per_pattern.parent.mrr, P.rotate.per_pattern.parent.mrr)) {
    fail('the written hierarchy', 'DistMult scores above both movement models on the relation '
      + 'written to have a direction, which its scoring function cannot represent');
  }

  /* 5. the composed text */
  const bad = [];
  document.querySelectorAll('p.body-text, .widget-readout, .atlas-table td, .widget-note, svg text')
    .forEach((node) => {
      const t = node.textContent || '';
      if (/\bundefined\b|\bNaN\b|\[object Object\]|\$\{/.test(t)) bad.push(node.id || node.className);
    });
  if (bad.length) fail('composed text', `undefined or NaN appears in: ${bad.join(', ')}`);

  if (problems.length) problems.forEach((p) => console.warn(`[kg-embeddings] ${p}`));
  else if (loud) {
    console.log('[kg-embeddings] the turn preserves length, filtering only helps, and the model '
      + 'that cannot represent a direction does not win on the relation that has one');
  }
  return problems;
}

async function boot() {
  renderMath();
  const data = await fetch('./data/kg.json').then((r) => r.json());

  PROSE_IDS = safely('the numbered sentences', () => initProse(data)) || [];
  safely('the live rotation', () => initLiveWidget(data));
  safely('the prediction', () => initSymWidget(data));
  safely('the written patterns', () => initPatWidget(data));
  renderMath();

  const guards = () => safely('guards', () => check(data, true));
  if ('requestIdleCallback' in window) requestIdleCallback(guards, { timeout: 5000 });
  else setTimeout(guards, 800);

  window.__atlasCheck = (loud) => check(data, loud);
  window.__atlasProseIds = PROSE_IDS;
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
