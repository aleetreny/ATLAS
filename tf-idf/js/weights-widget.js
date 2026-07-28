/* One document, weighted six ways, recomputed here from its integer counts.
 *
 * The point of the widget is that the three decisions are visible separately.
 * Switching the term-frequency mode reorders the bars; switching idf on and
 * off reorders them again, usually in the opposite direction; and the readout
 * says what the length normalisation did, which is the one you cannot see in
 * a ranking because it divides every bar by the same number.
 *
 * The document selector opens on a document that is neither the shortest nor
 * the longest, because the widget exists to be compared against its neighbours
 * and an extreme opens with nothing to compare.
 */
import { makeChart } from '../../assets/js/chart.js';
import { wordBars, seriesLabel } from '../../assets/js/textdraw.js';
import { vectorOf, topTerms, idfTable, idfOf } from './tfkit.js';

const TF_MODES = [
  { key: 'binary', label: 'present or not' },
  { key: 'raw', label: 'raw count' },
  { key: 'sublinear', label: '1 + log count' },
];

export function initWeightsWidget(data, ix) {
  const idf = idfTable(ix);
  let tf = 'sublinear';
  let useIdf = true;
  /* Documents worth landing on: long enough that the ranking has something to
   * say, and one from each of three different categories so the picker is not
   * three views of the same earnings report. Chosen by length percentile from
   * the shipped lengths, so this survives a regeneration. */
  const picks = choosePicks(ix);
  let pick = 0;

  const chart = makeChart('#weights-chart', {
    width: 640, height: 340, margin: { top: 34, right: 20, bottom: 20, left: 10 },
  });
  const { g, w, h } = chart;
  const readout = d3.select('#weights-readout');

  d3.select('#weights-tf').selectAll('button').data(TF_MODES).join('button')
    .attr('class', (d) => `atlas-btn${d.key === tf ? '' : ' ghost'}`)
    .text((d) => d.label)
    .on('click', (event, d) => { tf = d.key; sync(); render(); });

  d3.select('#weights-idf').selectAll('button')
    .data([{ key: true, label: 'weighted by rarity' }, { key: false, label: 'not weighted by rarity' }])
    .join('button')
    .attr('class', (d) => `atlas-btn${d.key === useIdf ? '' : ' ghost'}`)
    .text((d) => d.label)
    .on('click', (event, d) => { useIdf = d.key; sync(); render(); });

  const slider = d3.select('#weights-doc')
    .attr('min', 0).attr('max', picks.length - 1).attr('step', 1)
    .property('value', pick)
    .on('input', () => { pick = +slider.property('value'); render(); });

  function sync() {
    d3.select('#weights-tf').selectAll('button')
      .attr('class', (d) => `atlas-btn${d.key === tf ? '' : ' ghost'}`);
    d3.select('#weights-idf').selectAll('button')
      .attr('class', (d) => `atlas-btn${d.key === useIdf ? '' : ' ghost'}`);
  }

  function render() {
    const d = picks[pick];
    const vec = vectorOf(ix, d, { tf, useIdf, normalise: true, idf });
    const rows = topTerms(ix, vec, 10)
      .map((t) => ({ label: t.word, value: t.value, df: t.df }));
    g.selectAll('*').remove();
    const plot = g.append('g');
    wordBars(plot, rows, w, h - 6, {
      colour: 'var(--primary)',
      labelWidth: 118,
      valueFmt: (v) => v.toFixed(3),
      max: rows.length ? rows[0].value : 1,
    });
    seriesLabel(g, w, -14,
      `${ix.labels[d]} | ${ix.lens[d]} words | ${ix.ptr[d + 1] - ix.ptr[d]} in the vocabulary`,
      { anchor: 'end', size: 11 });
    seriesLabel(g, 0, -14, 'the ten heaviest terms', { anchor: 'start', size: 11 });

    d3.select('#weights-doc-label').text(`${pick + 1} of ${picks.length}`);

    /* The unnormalised length of this document under the current setting,
     * which is what the division actually removes. Two documents about the
     * same thing at different lengths have wildly different values here and
     * identical ones after. */
    const un = vectorOf(ix, d, { tf, useIdf, normalise: false, idf });
    const head = ix.titles[d] || '(no headline)';
    const top = rows[0];
    readout.html(
      `<div><span class="bold">${escapeHtml(head)}</span></div>`
      + `<div>Filed under <span class="bold">${ix.labels[d]}</span>. `
      + `Heaviest term <span class="bold">${escapeHtml(top.label)}</span> at `
      + `<span class="value">${top.value.toFixed(4)}</span>, and it appears in `
      + `<span class="value">${top.df.toLocaleString('en-US')}</span> of the `
      + `<span class="value">${ix.nTrain.toLocaleString('en-US')}</span> training documents, `
      + `so its rarity weight is <span class="value">${idfOf(top.df, ix.nTrain).toFixed(3)}</span>.</div>`
      + `<div>Before dividing by the length this document measures `
      + `<span class="value">${un.norm.toFixed(2)}</span>; after, every document on this page `
      + `measures exactly 1, which is the whole of what the third decision does.</div>`
    );
  }

  sync();
  render();
  return { render };
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

/* Six documents spread across the length range and across categories.
 * Deterministic from the shipped data, so it does not drift if the corpus is
 * rebuilt, and it deliberately avoids the two extremes: the shortest document
 * in this split has two words and its "ranking" is not a ranking. */
function choosePicks(ix) {
  const order = Array.from({ length: ix.docs }, (_, i) => i)
    .filter((i) => ix.ptr[i + 1] - ix.ptr[i] >= 12)
    .sort((a, b) => ix.lens[a] - ix.lens[b]);
  const out = [];
  const seen = new Set();
  [0.25, 0.45, 0.6, 0.75, 0.88, 0.96].forEach((q) => {
    let k = Math.min(order.length - 1, Math.floor(q * order.length));
    /* walk forward to the next unused category so the six are not all earnings */
    for (let step = 0; step < 200; step++) {
      const cand = order[Math.min(order.length - 1, k + step)];
      const key = ix.labels[cand];
      if (!seen.has(key) || step === 199) {
        out.push(cand);
        seen.add(key);
        return;
      }
    }
  });
  return out.length ? out : [0];
}
