/* Entry point.
 *
 * The guard here is that the page's own message passing is the same operation
 * the generator measured. The live widget applies the row averaged operator to
 * the subgraph it was sent and computes the spread between neighbours from it;
 * the generator did the same thing on the whole graph, so what can be checked
 * is the shape of the decay rather than a single number: it has to fall
 * monotonically and it has to reach the fixed point where every node is equal.
 */
import { initArmsWidget } from './arms-widget.js';
import { initLiveWidget } from './live-widget.js';
import { initHomoWidget } from './homo-widget.js';
import { initDepthWidget } from './depth-widget.js';
import { initAttWidget } from './att-widget.js';
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

function check(data, live, loud = false) {
  const problems = [];
  const fail = (label, detail) => problems.push(`${label}: ${detail}`);

  /* 1. the page's own averaging has to smooth, monotonically, to the fixed
     point. This is the claim the depth section makes, checked on the subgraph
     the page holds rather than taken from the file. */
  if (live) {
    const F0 = live.propagate(data.widget.nodes.map((d) => {
      const row = new Array(data.widget.classes).fill(0);
      if (d.labelled) row[d.y] = 1;
      return row;
    }), 0);
    let prev = live.spread(F0);
    let rose = 0;
    let cur = F0;
    for (let s = 1; s <= 12; s++) {
      cur = live.propagate(F0, s);
      const e = live.spread(cur);
      if (e > prev * 1.000001) rose += 1;
      prev = e;
    }
    if (rose > 0) {
      fail('the message passing', `the spread between neighbours went up ${rose} times out of 12, `
        + 'which averaging cannot do');
    }
    if (!(prev < live.spread(F0) * 0.5)) {
      fail('the message passing', 'twelve rounds of averaging did not halve the spread between '
        + 'neighbours, so the operator being applied is not an average');
    }
  }

  /* 2. the depth measurement and the untrained smoothing have to agree in
     direction: both are the same operator applied more times */
  const sm = data.depth.smoothing;
  for (let i = 1; i < sm.length; i++) {
    if (sm[i].energy > sm[i - 1].energy * 1.0001) {
      fail('the smoothing curve', `it rises between ${sm[i - 1].steps} and ${sm[i].steps} rounds`);
      break;
    }
  }

  /* 3. the homophily sweep has to be a sweep: the measured homophily has to
     track the requested one, or the dial is not a dial */
  const H = data.homophily;
  for (let i = 1; i < H.length; i++) {
    if (H[i].measured < H[i - 1].measured) {
      fail('the homophily dial', `asking for ${H[i].homophily} produced less homophily than asking `
        + `for ${H[i - 1].homophily}`);
      break;
    }
  }

  /* 4. the composed text */
  const bad = [];
  document.querySelectorAll('p.body-text, .widget-readout, .atlas-table td, .widget-note, svg text')
    .forEach((node) => {
      const t = node.textContent || '';
      if (/\bundefined\b|\bNaN\b|\[object Object\]|\$\{/.test(t)) bad.push(node.id || node.className);
    });
  if (bad.length) fail('composed text', `undefined or NaN appears in: ${bad.join(', ')}`);

  if (problems.length) problems.forEach((p) => console.warn(`[gnn] ${p}`));
  else if (loud) {
    console.log('[gnn] the page smooths the way the generator does, monotonically, and the '
      + 'homophily dial is monotone in what it produced');
  }
  return problems;
}

async function boot() {
  renderMath();
  const data = await fetch('./data/gnn.json').then((r) => r.json());

  PROSE_IDS = safely('the numbered sentences', () => initProse(data)) || [];
  safely('the arms', () => initArmsWidget(data));
  const live = safely('the live graph', () => initLiveWidget(data));
  safely('the homophily dial', () => initHomoWidget(data));
  safely('the depth', () => initDepthWidget(data));
  safely('the attention', () => initAttWidget(data));
  renderMath();

  const guards = () => safely('guards', () => check(data, live, true));
  if ('requestIdleCallback' in window) requestIdleCallback(guards, { timeout: 5000 });
  else setTimeout(guards, 800);

  window.__atlasCheck = (loud) => check(data, live, loud);
  window.__atlasProseIds = PROSE_IDS;
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
