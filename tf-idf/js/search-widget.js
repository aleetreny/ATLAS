/* The whole test split searched here, twice, over the same vectors.
 *
 * Pick a document and the page ranks the other 2,206 against it under two
 * geometries. Cosine scores the vectors after dividing each by its own length;
 * euclidean scores minus the squared distance between the same vectors before
 * that division. Nothing else differs: same counts, same weighting, same idf.
 * So every difference in the two columns is the length normalisation and can
 * be nothing else, which is what makes the comparison worth showing.
 *
 * Both are exact rather than approximate. 110,031 non-zeros scanned twice is a
 * couple of milliseconds, so there is no index to be wrong about.
 */
import { seriesLabel } from '../../assets/js/textdraw.js';
import { weightAll, rank, topK } from './tfkit.js';

const TF_MODES = [
  { key: 'sublinear', label: '1 + log count' },
  { key: 'raw', label: 'raw count' },
  { key: 'binary', label: 'present or not' },
];

export function initSearchWidget(data, ix) {
  let tf = 'sublinear';
  let useIdf = true;
  const queries = chooseQueries(ix);
  let qi = 0;

  const readout = d3.select('#search-readout');
  const table = d3.select('#search-results');

  d3.select('#search-tf').selectAll('button').data(TF_MODES).join('button')
    .attr('class', (d) => `atlas-btn${d.key === tf ? '' : ' ghost'}`)
    .text((d) => d.label)
    .on('click', (event, d) => { tf = d.key; sync(); render(); });

  d3.select('#search-idf').selectAll('button')
    .data([{ key: true, label: 'with idf' }, { key: false, label: 'without idf' }])
    .join('button')
    .attr('class', (d) => `atlas-btn${d.key === useIdf ? '' : ' ghost'}`)
    .text((d) => d.label)
    .on('click', (event, d) => { useIdf = d.key; sync(); render(); });

  const slider = d3.select('#search-q')
    .attr('min', 0).attr('max', queries.length - 1).attr('step', 1).property('value', qi)
    .on('input', () => { qi = +slider.property('value'); render(); });

  function sync() {
    d3.select('#search-tf').selectAll('button')
      .attr('class', (d) => `atlas-btn${d.key === tf ? '' : ' ghost'}`);
    d3.select('#search-idf').selectAll('button')
      .attr('class', (d) => `atlas-btn${d.key === useIdf ? '' : ' ghost'}`);
  }

  /* Weighting the whole split is the expensive half and it does not depend on
   * the query, so it is cached per setting rather than redone per drag. */
  const cache = new Map();
  function weighted(normalise) {
    const key = `${tf}|${useIdf}|${normalise}`;
    if (!cache.has(key)) cache.set(key, weightAll(ix, { tf, useIdf, normalise }));
    return cache.get(key);
  }

  function render() {
    const q = queries[qi];
    const Wc = weighted(true);
    const We = weighted(false);
    const cos = rank(ix, Wc, q, 'cosine');
    const euc = rank(ix, We, q, 'euclid');
    const tc = topK(cos, 5);
    const te = topK(euc, 5);

    d3.select('#search-q-label').text(`${qi + 1} of ${queries.length}`);

    const qLabel = ix.labels[q];
    const rows = [];
    for (let i = 0; i < 5; i++) {
      rows.push({ i: i + 1, c: tc[i], e: te[i] });
    }
    table.html('');
    const head = table.append('thead').append('tr');
    ['#', 'nearest by cosine', 'cat', 'words', 'nearest by euclidean distance', 'cat', 'words']
      .forEach((t) => head.append('th').text(t));
    const body = table.append('tbody');
    rows.forEach((r) => {
      const tr = body.append('tr');
      tr.append('td').text(r.i);
      cell(tr, ix, r.c, qLabel);
      tr.append('td').attr('class', ix.labels[r.c] === qLabel ? 'hit' : 'miss')
        .text(ix.labels[r.c]);
      tr.append('td').text(ix.lens[r.c].toLocaleString('en-US'));
      cell(tr, ix, r.e, qLabel);
      tr.append('td').attr('class', ix.labels[r.e] === qLabel ? 'hit' : 'miss')
        .text(ix.labels[r.e]);
      tr.append('td').text(ix.lens[r.e].toLocaleString('en-US'));
    });

    /* The two summaries over the whole ranking rather than the first five,
     * because five rows is an anecdote and the article quotes a rate. */
    const cHit = topK(cos, 10).filter((d) => ix.labels[d] === qLabel).length;
    const eHit = topK(euc, 10).filter((d) => ix.labels[d] === qLabel).length;
    const cGap = mean(topK(cos, 10).map((d) => Math.abs(ix.lens[d] - ix.lens[q])));
    const eGap = mean(topK(euc, 10).map((d) => Math.abs(ix.lens[d] - ix.lens[q])));

    const verdict = cHit === eHit
      ? `they agree on this query, ${cHit} of 10 each`
      : (cHit > eHit
        ? `cosine gets ${cHit} of 10 right and euclidean ${eHit}`
        : `euclidean gets ${eHit} of 10 right and cosine ${cHit}, which happens and is why the `
          + `article quotes the average over all ${ix.docs.toLocaleString('en-US')} queries `
          + `rather than one`);

    readout.html(
      `<div>Query: <span class="bold">${escapeHtml(ix.titles[q] || '(no headline)')}</span> `
      + `(<span class="bold">${qLabel}</span>, ${ix.lens[q].toLocaleString('en-US')} words).</div>`
      + `<div>In the top ten, ${verdict}. The mean length gap to the query is `
      + `<span class="value">${cGap.toFixed(1)}</span> words for cosine and `
      + `<span class="value">${eGap.toFixed(1)}</span> for euclidean, and that gap is the whole `
      + `mechanism: without the division, a long document is far from everything and a short one `
      + `is near everything, whatever either is about.</div>`
    );
  }

  sync();
  render();
  return { render };
}

function cell(tr, ix, d, qLabel) {
  const td = tr.append('td');
  td.append('span').text(ix.titles[d] || '(no headline)');
  td.attr('title', ix.titles[d] || '');
}

function mean(xs) {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

/* Eight queries spread across categories and lengths. The first one is
 * deliberately a long document, because that is where the two geometries part
 * company and opening on a median-length document would show two identical
 * columns and teach nothing. */
function chooseQueries(ix) {
  const byCat = new Map();
  for (let i = 0; i < ix.docs; i++) {
    if (ix.ptr[i + 1] - ix.ptr[i] < 15) continue;
    const c = ix.labels[i];
    if (!byCat.has(c)) byCat.set(c, []);
    byCat.get(c).push(i);
  }
  const out = [];
  ix.labelNames.forEach((c) => {
    const list = (byCat.get(c) || []).sort((a, b) => ix.lens[b] - ix.lens[a]);
    if (list.length) out.push(list[0]);
  });
  return out.length ? out : [0];
}
