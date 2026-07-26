/* Entry point: render math, load data, boot every widget in isolation. */
import { initScrollyMerge } from './scrolly-merge.js';
import { initDendroWidget } from './dendro-widget.js';
import { initChainWidget } from './chain-widget.js';
import { initBirchWidget } from './birch-widget.js';
import { agglomerate, cutTree, ari } from './hierkit.js';

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

/* One widget throwing must not take the rest of the page down with it. */
function safely(name, fn) {
  try {
    fn();
  } catch (err) {
    console.error(`widget "${name}" failed to initialise:`, err);
  }
}

async function boot() {
  renderMath();
  const data = await fetch('./data/trees.json').then((r) => r.json());
  safely('merge scrolly', () => initScrollyMerge());
  safely('dendrogram', () => initDendroWidget(data));
  safely('the bridge', () => initChainWidget(data));
  safely('birch', () => initBirchWidget(data));

  /* Runtime checks, deferred so boot stays quick: the browser's own ward
   * agglomeration of all 178 wines must reproduce scipy's tree (same merge
   * heights, identical k=3 cut), and its trees on the bridge must match the
   * shipped scipy cuts for every linkage. */
  const check = () => {
    const Zjs = agglomerate(data.wine.features, 'ward');
    const Zsp = data.wine.linkages.ward.Z;
    let worst = 0;
    for (let i = 0; i < Zjs.length; i++) {
      worst = Math.max(worst, Math.abs(Zjs[i][2] - Zsp[i][2]) / Math.max(Zsp[i][2], 1e-9));
    }
    if (worst > 1e-3) console.warn(`browser ward heights diverge from scipy by up to ${(worst * 100).toFixed(2)}%`);
    const aJs = cutTree(Zjs, 178, 3);
    const aSp = cutTree(Zsp, 178, 3);
    if (ari(aJs, aSp) < 0.999) console.warn('browser ward k=3 cut disagrees with scipy');

    for (const m of ['ward', 'complete', 'average', 'single']) {
      const Zc = agglomerate(data.chain.points, m);
      const mine = cutTree(Zc, data.chain.points.length, 2);
      const theirs = data.chain.linkages[m].labels_k2.map((v) => v - 1);
      if (ari(mine, theirs) < 0.999) console.warn(`bridge ${m} k=2 cut disagrees with scipy`);
    }
  };
  if ('requestIdleCallback' in window) requestIdleCallback(check, { timeout: 4000 });
  else setTimeout(check, 800);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
