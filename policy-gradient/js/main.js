/* Entry point.
 *
 * The guards recompute rather than reread: the policy from its parameters, its
 * values and its exact gradient by solving two linear systems, and, on the deep
 * pass, the estimator itself over four thousand episodes rolled here, so the
 * unbiasedness the whole article rests on is confirmed in the browser.
 */
import { initGradScrolly } from './scrolly-grad.js';
import { initNoiseWidget } from './noise-widget.js';
import { initTrainWidget } from './train-widget.js';
import { initProse } from './prose.js';
import { makeWorld, softmaxRows, exactValues, exactGradient, sampledGradient, norm, cosine }
  from './pgkit.js';

function renderMath() {
  if (typeof renderMathInElement !== 'function') return;
  renderMathInElement(document.body, {
    delimiters: [
      { left: '$$', right: '$$', display: true },
      { left: '\\(', right: '\\)', display: false },
      { left: '$', right: '$', display: false },
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
  if (!ok) console.warn(`[policy-gradient] ${label}: ${detail}`);
  return ok;
}

function runGuards(data, deep) {
  const O = data.optimal;
  const world = makeWorld({ size: O.size, gamma: O.gamma, goal: O.goal, trap: O.trap,
    stepCost: O.step_cost, goalReward: O.goal_reward, trapReward: O.trap_reward });
  const G = data.gradient;
  const pi = softmaxRows(G.theta, world.nS, world.nA);

  /* 1. the policy the file recorded, recomputed from its parameters */
  let worstPi = 0;
  G.policy.forEach((row, s) => row.forEach((v, a) => {
    worstPi = Math.max(worstPi, Math.abs(v - pi[s][a]));
  }));
  check('the policy', worstPi < 2e-5, `a probability moves by ${worstPi}`);

  /* 2. the values of that policy, from solving the linear system here */
  const { V } = exactValues(world, pi);
  let worstV = 0;
  G.V.forEach((v, s) => { worstV = Math.max(worstV, Math.abs(v - V[s])); });
  check('the values', worstV < 2e-5, `a value moves by ${worstV}`);
  check('what the policy is worth', Math.abs(V[O.start] - G.J) < 2e-5,
    `${V[O.start]} here against ${G.J} in the file`);

  /* 3. the exact gradient, recomputed, and its length */
  const { grad } = exactGradient(world, pi, O.start);
  check('the gradient', Math.abs(norm(grad) - G.exact_norm) < 2e-4,
    `length ${norm(grad).toFixed(6)} here against ${G.exact_norm}`);

  /* 4. the estimator is unbiased: the average of many lands on the gradient,
   *    which is the one claim the whole article rests on, so the browser rolls
   *    its own episodes rather than repeating the number the file reports */
  if (deep) {
    let seed = 20260729;
    const rand = () => {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      return seed / 4294967296;
    };
    const acc = new Array(grad.length).fill(0);
    const n = 4000;
    for (let i = 0; i < n; i++) {
      const g = sampledGradient(world, pi, O.start, rand, 1, null);
      for (let k = 0; k < acc.length; k++) acc[k] += g[k] / n;
    }
    const cos = cosine(acc, grad);
    check('unbiased', cos > 0.95 && norm(acc) > norm(grad) * 0.8
      && norm(acc) < norm(grad) * 1.25,
    `averaging ${n} estimates here gives cosine ${cos.toFixed(4)} and length `
      + `${norm(acc).toFixed(5)} against ${norm(grad).toFixed(5)}`);
    check('the same in the file', G.bias.plain.cosine > 0.98 && G.bias.baseline.cosine > 0.98,
      `the generator reported cosine ${G.bias.plain.cosine} plain and `
      + `${G.bias.baseline.cosine} with a baseline`);
  }

  /* 5. the baseline never makes the estimate worse, at any batch size */
  let worse = 0;
  G.rows.filter((d) => d.kind === 'plain').forEach((d) => {
    const b = G.rows.find((q) => q.kind === 'baseline' && q.episodes === d.episodes);
    if (b.error > d.error) worse += 1;
  });
  check('the baseline', worse === 0,
    `${worse} batch sizes where subtracting the value made the estimate further off`);

  /* 6. clipping only ever touches the runs that reuse a batch */
  if (deep) {
    check('clipping', data.training.reinforce.clipped === 0
      && data.training.ppo.clipped > 0,
    'the clip fraction is not zero on the single pass runs, which cannot happen');
  }

  /* 7. nothing may read undefined or NaN */
  const empty = [];
  document.querySelectorAll('p.body-text, .widget-readout, .atlas-table td, .gen-note, svg text')
    .forEach((node) => {
      const t = node.textContent || '';
      if (/\bundefined\b|\bNaN\b|\[object Object\]/.test(t)) {
        empty.push(node.id || node.className || node.tagName);
      }
    });
  check('composed text', empty.length === 0, `undefined or NaN appears in: ${empty.join(', ')}`);

  console.info(`[policy-gradient] load-time checks complete: policy within `
    + `${worstPi.toExponential(1)}, values within ${worstV.toExponential(1)}, gradient length `
    + `${norm(grad).toFixed(5)}`);
  return { worstPi, worstV };
}

async function boot() {
  renderMath();
  const data = await fetch('./data/pg.json').then((res) => res.json());

  safely('prose', () => initProse(data));
  safely('the gradient', () => initGradScrolly(data));
  safely('the noise', () => initNoiseWidget(data));
  safely('four runs', () => initTrainWidget(data));

  renderMath();

  window.__atlasCheck = (deep = false) => safely('guards', () => runGuards(data, deep));
  const guards = () => window.__atlasCheck(false);
  if ('requestIdleCallback' in window) requestIdleCallback(guards, { timeout: 5000 });
  else setTimeout(guards, 1000);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
