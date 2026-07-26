/* Entry point: render math, load data, boot every widget in isolation. */
import { initScrollyDensity } from './scrolly-density.js';
import { initLabWidget } from './lab-widget.js';
import { initKdistWidget } from './kdist-widget.js';
import { initTrapWidget } from './trap-widget.js';
import { dbscan, ari } from './dbkit.js';

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
  const data = await fetch('./data/density.json').then((r) => r.json());
  safely('density scrolly', () => initScrollyDensity(data));
  safely('two-dial lab', () => initLabWidget(data));
  safely('k-distance', () => initKdistWidget(data));
  safely('the trap', () => initTrapWidget(data));

  /* Runtime check: this page's DBSCAN must reproduce every scikit-learn
   * reference answer shipped with the data: cluster count, noise count and
   * ARI, at every (eps, minPts) pair. */
  for (const [name, ds] of Object.entries(data.datasets)) {
    for (const ref of data.refs[name]) {
      const r = dbscan(ds.points, ref.eps, ref.min_samples);
      const a = ari(r.labels, ds.true_labels);
      if (r.nClusters !== ref.n_clusters || r.nNoise !== ref.n_noise || Math.abs(a - ref.ari) > 0.002) {
        console.warn(`DBSCAN mismatch on ${name} eps ${ref.eps} minPts ${ref.min_samples}: ` +
          `browser ${r.nClusters}/${r.nNoise}/${a.toFixed(3)} vs sklearn ${ref.n_clusters}/${ref.n_noise}/${ref.ari}`);
      }
    }
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
