/* Entry point.
 *
 * The reuse claim is the one that has to hold before anything else on this page
 * means anything, so the guard checks it first and checks it against numbers
 * the other two articles published rather than against something this one
 * computed. After that it checks the directions, because the whole article is a
 * comparison between two tables and a sentence that got a direction wrong would
 * be the easiest thing here to ship.
 */
import { initProse, PROSE_IDS } from './prose.js';
import { initScoreWidget } from './score-widget.js';
import { initSanityWidget } from './sanity-widget.js';
import { initAttWidget } from './att-widget.js';

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

function check(data) {
  const problems = [];
  const M = data.meta;
  const S = Object.fromEntries(data.scores.rows.map((r) => [r.key, r]));
  const full = data.sanity[data.sanity.length - 1];

  /* ---- the reuse, which everything else stands on */
  if (!M.reuse.matches) {
    problems.push('the rebuilt scenes do not match the published sprite, so these are not '
      + 'the detection article pictures');
  }
  if (!(M.reuse.worst_box < 0.01)) {
    problems.push(`the numpy detector disagrees with the published boxes by `
      + `${M.reuse.worst_box} pixels`);
  }
  if (!(data.attention.demo_acc > 0.85)) {
    problems.push(`the rebuilt encoder tags at ${data.attention.demo_acc}, so it is not the `
      + 'attention article model');
  }

  /* ---- Grad-CAM is CAM on a one by one head, which is an identity */
  if (!(data.identity.worst_relative < 1e-6)) {
    problems.push(`Grad-CAM and the class activation map are supposed to coincide and differ `
      + `by ${data.identity.worst_relative}`);
  }

  /* ---- every method has to beat the noise control, or something is broken
     rather than interesting */
  ['gradient', 'grad_input', 'gradcam', 'occlusion'].forEach((k) => {
    if (!(S[k].pointing > S.random.pointing)) {
      problems.push(`${k} does not beat smooth noise, which means it is broken and not that `
        + 'it is uninformative');
    }
  });

  /* ---- the two claims the article turns on */
  if (!(S.input.overlap > S.gradient.overlap)) {
    problems.push('the page says the raw input beats the plain gradient on overlap, and the '
      + 'table disagrees');
  }
  if (!(Math.abs(full.gradcam) < 0.4 && full.gradient > 0.8 && full.grad_input > 0.8)) {
    problems.push('the randomisation test is supposed to break Grad-CAM and leave the '
      + `gradient methods standing, and it gives ${full.gradcam}, ${full.gradient} and `
      + `${full.grad_input}`);
  }
  const D = data.depth;
  if (!(D[D.length - 1].pointing > D[0].pointing)) {
    problems.push('the page says the deep feature map localises better than the shallow one, '
      + 'and the ladder disagrees');
  }

  /* ---- attention */
  const A = data.attention;
  if (A.adversarial && !(A.adversarial.tv > 0.5 && A.adversarial.same === 1)) {
    problems.push(`the adversarial attention is supposed to be far away and harmless, and it `
      + `is ${A.adversarial.tv} away with ${A.adversarial.same} of the tags unchanged`);
  }
  /* ---- la sección de atención descansa en dos cosas y ninguna es que las dos
     ablaciones coincidan: la buena es la publicada sobre el test entero, y la
     de ocho frases está en la página como ejemplo de lo que mide un conjunto
     demasiado pequeño. Así que se comprueba justo eso. */
  if (A.rows.some((row) => row.published === null || row.published === undefined)) {
    problems.push('some head has no published drop, so the importance axis has no source');
  }
  const step = 1 / A.tokens;
  /* Los valores viajaron redondeados a seis decimales, así que multiplicar por
     95 los aleja del entero hasta 5e-5: con 1e-6 el guardia disparaba sobre
     datos correctos. */
  if (!A.rows.every((row) => Math.abs(row.ablation * A.tokens
      - Math.round(row.ablation * A.tokens)) < 1e-3)) {
    problems.push(`the eight sentence ablation is supposed to be quantised in steps of `
      + `1/${A.tokens} and it is not, so the resolution argument does not hold`);
  }
  const spread = Math.max(...A.rows.map((r) => r.published))
    - Math.min(...A.rows.map((r) => r.published));
  if (!(spread < 4 * step)) {
    problems.push(`the published drops span ${spread.toFixed(4)}, which is more than four `
      + `marks of a ${step.toFixed(4)} ruler, and the page says they are crowded inside it`);
  }
  if (!(Math.abs(A.corrs.peak) < Math.abs(A.corrs_small.peak))) {
    problems.push(`the page says the small sample invents a correlation, and peakedness `
      + `ranks at ${A.corrs.peak} against the published drop and ${A.corrs_small.peak} `
      + `against the small one`);
  }
  const peakiest = [...A.rows].sort((a, b) => b.peak - a.peak)[0];
  if (!(peakiest.layer === A.top_head.layer && peakiest.head === A.top_head.head)) {
    problems.push('the page opens that section by saying the most peaked head is also the '
      + 'most important, and it is not');
  }

  if (problems.length) {
    console.warn(`this page disagrees with its references in ${problems.length} place(s):`);
    problems.forEach((p) => console.warn(`  ${p}`));
  } else {
    console.log('the rebuilt models reproduce what the detection and attention articles '
      + 'published, Grad-CAM equals the class activation map, and every direction the prose '
      + 'states matches its table');
  }
  return problems;
}

async function boot() {
  const data = await fetch('./data/saliency.json').then((r) => r.json());
  safely('the numbered sentences', () => initProse(data));
  safely('six maps and two controls', () => initScoreWidget(data));
  safely('the test that changes the ranking', () => initSanityWidget(data));
  safely('the other picture everybody shows', () => initAttWidget(data));
  renderMath();

  if (window.requestIdleCallback) {
    window.requestIdleCallback(() => check(data));
  } else {
    setTimeout(() => check(data), 400);
  }
  window.__atlasCheck = () => check(data);
  window.__atlasProseIds = PROSE_IDS;
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
