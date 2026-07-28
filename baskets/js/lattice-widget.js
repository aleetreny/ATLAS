/* Every subset, and the ones Apriori never looks at.
 *
 * The lattice is enumerated in the page, exactly, so the two numbers in the
 * readout are both measured here: how many subsets there are, and how many the
 * algorithm counted. The claim that pruning loses nothing is checked the same
 * way, by comparing the sets Apriori found against the sets exhaustion found,
 * rather than by citing the theorem.
 */
import { makeChart } from '../../assets/js/chart.js';
import { enumerateAll, apriori } from './basketkit.js';
import { latticeLayout, legend, seriesLabel } from './draw.js';

const KS = [6, 8, 10, 12];

export function initLatticeWidget(data, db) {
  const G = data.groceries;
  const SUPPORTS = G.ship_supports;
  let ki = 2;                    /* ten items: 1,023 subsets, enough to look
                                    hopeless and few enough to draw */
  let si = SUPPORTS.indexOf(0.02) >= 0 ? SUPPORTS.indexOf(0.02) : 2;

  const chart = makeChart('#lattice-chart', {
    width: 640, height: 420, margin: { top: 56, right: 26, bottom: 46, left: 46 },
  });
  const { g, w, h } = chart;
  const readout = d3.select('#lattice-readout');
  const sSlider = d3.select('#lattice-support')
    .attr('min', 0).attr('max', SUPPORTS.length - 1).attr('step', 1).property('value', si);

  d3.select('#lattice-ks').selectAll('button').data(KS).join('button')
    .attr('class', (d, i) => `atlas-btn${i === ki ? '' : ' ghost'}`)
    .text((d) => `${d} items`)
    .on('click', (event, d) => {
      ki = KS.indexOf(d);
      d3.select('#lattice-ks').selectAll('button')
        .attr('class', (q, i) => `atlas-btn${i === ki ? '' : ' ghost'}`);
      render();
    });

  /* every lattice, once: 2^12 is four thousand nodes and recomputing it on
     every drag of the support slider would be four thousand nodes per frame
     for no reason, because the supports do not depend on the threshold */
  const CACHE = KS.map((k) => ({ k, all: enumerateAll(db, k) }));

  function render() {
    si = +sSlider.property('value');
    const sup = SUPPORTS[si];
    const minCount = Math.ceil(sup * db.n);
    const { k, all } = CACHE[ki];
    const keys = [...all.keys()];

    const ap = apriori(db, k, minCount);
    const counted = new Set();
    ap.levels.forEach((lv) => lv.forEach((e) => counted.add(e.key.join(','))));
    const frequent = new Set(counted);
    /* Apriori also counts candidates that turn out infrequent; mark them by
       re-running the join and keeping what it counted but did not keep */
    const brute = new Set(keys.filter((q) => all.get(q) >= minCount));
    const same = brute.size === frequent.size && [...frequent].every((q) => brute.has(q));

    const { pos } = latticeLayout(keys, k, w, h, { top: 10, bottom: 10 });
    g.selectAll('*').remove();

    const marks = keys.map((key) => ({
      key,
      ...pos.get(key),
      freq: all.get(key) >= minCount,
      count: all.get(key),
    }));
    const r = k <= 8 ? 4.6 : k <= 10 ? 3 : 1.9;
    g.selectAll('circle.node').data(marks).join('circle').attr('class', 'node')
      .attr('cx', (d) => d.x).attr('cy', (d) => d.y).attr('r', r)
      .attr('fill', (d) => (d.freq ? 'var(--primary)' : 'var(--stone)'))
      .attr('opacity', (d) => (d.freq ? 0.95 : 0.5))
      .attr('stroke', 'var(--paper)').attr('stroke-width', 0.5);

    /* the size axis on the left, which is the only axis this picture has */
    for (let size = 1; size <= k; size++) {
      const y = 10 + ((h - 20) * (size - 1)) / Math.max(1, k - 1);
      seriesLabel(g, -10, y + 4, String(size), { anchor: 'end', size: 10.5, opacity: 0.75 });
    }
    seriesLabel(g, -10, -12, 'items in the set', { anchor: 'end', size: 10.5, opacity: 0.75 })
      .attr('x', -34).attr('text-anchor', 'start');

    legend(g, [
      { label: `frequent (${brute.size})`, shape: 'dot', colour: 'var(--primary)' },
      { label: `not frequent (${keys.length - brute.size})`, shape: 'dot', colour: 'var(--stone)' },
    ], { x0: 2, y0: -32, gap: 220 });

    d3.select('#lattice-support-label').text(`${(100 * sup).toFixed(2)}%`);
    const skipped = keys.length - ap.work.counted;
    readout.html(
      `<span class="slider-label">minimum support <span class="value">`
      + `${(100 * sup).toFixed(2)}%</span></span> `
      + `<span class="slider-label">items <span class="value">${k}</span></span>. `
      + `The ${k} most common items have <span class="bold">`
      + `${keys.length.toLocaleString('en-US')}</span> non-empty subsets, and this page has just `
      + `counted the support of every one of them. `
      + `<span class="bold">${brute.size.toLocaleString('en-US')}</span> of them reach `
      + `${minCount} baskets. `
      + `Apriori, run here on the same data, counted `
      + `<span class="bold">${ap.work.counted.toLocaleString('en-US')}</span> candidates and threw `
      + `away ${ap.work.pruned.toLocaleString('en-US')} more without counting them, so it never `
      + `looked at ${skipped.toLocaleString('en-US')} of the `
      + `${keys.length.toLocaleString('en-US')} subsets, which is `
      + `${(100 * skipped / keys.length).toFixed(2)}% of the lattice. `
      + (same
        ? `<span class="bold">It found exactly the same sets as the exhaustive count.</span> Not `
          + `nearly the same: the same, checked set by set on this page every time you move a `
          + `control.`
        : `<span class="bold">It found a different set from the exhaustive count, which cannot `
          + `happen and means something on this page is wrong.</span>`)
      + ` These figures are for the ${k} items the buttons picked out of the `
      + `${db.names.length} this page ships, which is as far as a browser can enumerate one `
      + `subset at a time; the tables below use all ${G.all_names.length}.`
    );
  }

  sSlider.on('input', render);
  render();
  return { render };
}
