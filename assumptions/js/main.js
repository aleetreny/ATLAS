/* Entry point: load the data, boot every widget in isolation, and then check
 * this page's four algorithms against the four reference implementations the
 * generator ran. A page that claims to run PAM and runs something else is
 * worse than a page with a picture of PAM, so the checks are loud.
 */
import { initProse } from './prose.js';
import { initScrollySwap } from './scrolly-swap.js';
import { initOutlierWidget } from './outlier-widget.js';
import { initWineWidget } from './wine-widget.js';
import { initModesWidget } from './modes-widget.js';
import { initOpticsWidget } from './optics-widget.js';
import { initShiftWidget } from './shift-widget.js';
import { distanceMatrix, pam, kmodes, optics, extractDbscan, meanShift, kmeans, ari } from './relaxkit.js';

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

/* `deep` runs everything, which takes a few seconds and is what a verification
 * pass wants. The automatic run at load time checks one thing per algorithm,
 * which is enough to catch a page whose code and whose references have drifted
 * apart, and cheap enough not to be felt. */
function check(data, deep = false) {
  const problems = [];
  const some = (arr, n) => (deep ? arr : arr.slice(0, n));

  /* PAM against the Rust FasterPAM, on the blobs and on all three wine metrics */
  const blobs = data.datasets.blobs.points;
  const pb = pam(distanceMatrix(blobs), 3);
  if (Math.abs(pb.loss - data.medoids.loss) > 0.005) {
    problems.push(`PAM on the blobs: ${pb.loss.toFixed(4)} here against ${data.medoids.loss} offline`);
  }
  some(data.wine.metrics, 1).forEach((m) => {
    const r = pam(distanceMatrix(data.wine.features, m.key), 3);
    if (Math.abs(r.loss - m.loss) > 0.01) {
      problems.push(`PAM on the wine (${m.key}): ${r.loss.toFixed(3)} here against ${m.loss} offline`);
    }
  });

  /* k-modes against the kmodes package's Cao-initialised run */
  const km = kmodes(data.modes.codes, 3);
  if (km.cost !== data.modes.kmodes.cost) {
    problems.push(`k-modes cost ${km.cost} here against ${data.modes.kmodes.cost} offline`);
  }
  /* and against its own invariance claim, which is the section's whole point */
  some(data.modes.perms.slice(0, 3), 1).forEach((P, i) => {
    const permuted = data.modes.codes.map((row) => row.map((c, j) => P[j][c]));
    const kmP = kmodes(permuted, 3);
    if (kmP.cost !== km.cost || ari(km.labels, kmP.labels) < 0.9999) {
      problems.push(`k-modes changed its answer under relabelling ${i + 1}`);
    }
    /* and the other half of the claim: that k-means on the codes does move,
       to the value the offline sweep recorded for this very permutation */
    const a = ari(data.wine.true, kmeans(permuted.map((r) => r.map(Number)), 3, { restarts: 30 }).labels);
    if (Math.abs(a - data.modes.perm_rows[i].km) > 0.002) {
      problems.push(`k-means on relabelling ${i + 1}: ARI ${a.toFixed(3)} here against `
        + `${data.modes.perm_rows[i].km} offline`);
    }
  });

  /* OPTICS against scikit-learn: the drawn curve, not the ordering, because
     points whose reachability ties may legitimately be visited either way */
  some(Object.entries(data.optics), 1).forEach(([name, ref]) => {
    const r = optics(data.datasets[name].points, data.meta.min_samples);
    let worst = 0;
    r.order.forEach((p, i) => {
      const mine = r.reach[p];
      const theirs = ref.reachability[i];
      if (theirs === null) return;
      if (Number.isFinite(mine)) worst = Math.max(worst, Math.abs(mine - theirs));
    });
    /* the reference is written with six decimals, so half of the last one is
       as sharp as this comparison can be */
    if (worst > 1e-6) problems.push(`OPTICS on the ${name}: reachability differs by ${worst.toExponential(2)}`);
    /* and the extraction, which is what the widget actually draws: a handful of
       radii spread across the sweep, against scikit-learn's own answers */
    some([0.15, 0.4, 0.7], 1).forEach((frac) => {
      const cut = ref.cuts[Math.floor(ref.cuts.length * frac)];
      const labels = extractDbscan(r.order, r.reach, r.core, cut.eps);
      const nC = Math.max(-1, ...labels) + 1;
      const nN = labels.filter((l) => l === -1).length;
      const a = ari(data.datasets[name].true_labels, labels);
      if (nC !== cut.n_clusters || nN !== cut.n_noise || Math.abs(a - cut.ari) > 0.002) {
        problems.push(`OPTICS cut on the ${name} at eps ${cut.eps}: ${nC}/${nN}/${a.toFixed(3)} here `
          + `against ${cut.n_clusters}/${cut.n_noise}/${cut.ari} offline`);
      }
    });
  });

  /* mean shift against scikit-learn, at the bandwidths the generator checked */
  some(Object.entries(data.shift), 1).forEach(([name, ref]) => {
    const pts = data.datasets[name].points;
    some(ref.sweep.filter((r) => [1.0, 1.5, 2.0, 2.5, 3.0].includes(r.bw)), 2).forEach((r) => {
      const res = meanShift(pts, r.bw);
      const a = ari(data.datasets[name].true_labels, res.labels);
      if (res.centers.length !== r.modes || Math.abs(a - r.ari) > 0.002) {
        problems.push(`mean shift on the ${name} at bandwidth ${r.bw}: `
          + `${res.centers.length} modes / ARI ${a.toFixed(3)} here against ${r.modes} / ${r.ari} offline`);
      }
    });
  });

  /* and k-means, which is the control every comparison leans on */
  const sample = some([0, 40, 74, 120, 200, 300], 2).filter((i) => i < data.outlier.grid.length);
  sample.forEach((i) => {
    const d = data.outlier.grid[i];
    const pts = blobs.concat([[data.outlier.origin[0] + data.outlier.dir[0] * d,
      data.outlier.origin[1] + data.outlier.dir[1] * d]]);
    const r = kmeans(pts, 3, { restarts: 20 });
    const a = ari(data.datasets.blobs.true_labels, r.labels.slice(0, 300));
    if (Math.abs(a - data.outlier.sweep.km[i]) > 0.002) {
      problems.push(`k-means at d = ${d}: ARI ${a.toFixed(3)} here against ${data.outlier.sweep.km[i]} offline`);
    }
  });

  if (problems.length) {
    console.warn(`this page disagrees with its references in ${problems.length} place(s):`);
    problems.forEach((p) => console.warn(`  ${p}`));
  } else {
    console.log(`every algorithm on this page matches its reference implementation`
      + `${deep ? '' : ' (spot check; window.__atlasCheck(true) runs all of them)'}`);
  }
  return problems;
}

async function boot() {
  const data = await fetch('./data/relaxations.json').then((r) => r.json());
  safely('the numbered sentences', () => initProse(data));
  safely('pam scrolly', () => initScrollySwap(data));
  safely('the outlier', () => initOutlierWidget(data));
  safely('the wine exemplars', () => initWineWidget(data));
  safely('categories', () => initModesWidget(data));
  safely('optics', () => initOpticsWidget(data));
  safely('mean shift', () => initShiftWidget(data));
  renderMath();
  /* after the page is interactive: the checks take a second and nothing on
     screen depends on them */
  if (window.requestIdleCallback) {
    window.requestIdleCallback(() => check(data));
  } else {
    setTimeout(() => check(data), 400);
  }
  window.__atlasCheck = (deep = false) => check(data, deep);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
