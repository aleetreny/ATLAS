/* Every number in the running text, composed from the file at load. */
const n0 = (v) => Number(v).toLocaleString('en-US');
const f4 = (v) => Number(v).toFixed(4);
const pct = (v) => `${(v * 100).toFixed(1)}%`;

export const PROSE_IDS = [
  'opening-note', 'filter-intro', 'filter-note', 'churn-intro', 'churn-note',
  'hybrid-intro', 'hybrid-note', 'bytes-intro', 'bytes-note', 'verdict-intro',
  'verdict-filter', 'verdict-filter-fix', 'verdict-churn', 'verdict-churn-fix',
  'verdict-hybrid', 'verdict-hybrid-fix', 'verdict-bytes', 'verdict-bytes-fix',
  'closing-1', 'closing-2', 'ref-note',
];

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

export function initProse(data) {
  const M = data.meta;
  const F = data.filter;
  const C = data.churn;
  const H = data.hybrid;
  const B = data.bytes;

  const rows = F.rows.slice().sort((a, b) => a.selectivity - b.selectivity);
  const tight = rows[0];
  const loose = rows[rows.length - 1];
  /* the loosest filter that has ALREADY broken the subgraph, which is the
     threshold the sentence is about. Scanning from rows[0] finds the tightest
     one instead, which is true of every row past the split and says nothing */
  const split = rows.slice().reverse().find((r) => r.pieces > 1);
  const peak = rows.reduce((a, b) => (b.pieces > a.pieces ? b : a));
  const cats = F.categories.slice().sort((a, b) => a.selectivity - b.selectivity);
  const worstCat = cats[0];
  const churn = C.rows;
  const deep = churn[churn.length - 1];
  const pair = churn.filter((r) => r.rebuilt_recall != null).pop();
  const short = H.rows[0];
  const long = H.rows[H.rows.length - 1];
  const flips = Math.sign(short.dense - short.sparse) !== Math.sign(long.dense - long.sparse);
  const float = B.rows[0];
  const smallest = B.rows.reduce((a, b) => (b.bytes < a.bytes ? b : a));
  const rer = B.rerank;
  const enough = rer.find((r) => r.recall >= 0.95) || rer[rer.length - 1];

  set('opening-note',
    `Same collection as the <a href="../ann-search/">approximate search article</a>, and same `
    + `answer key: `
    + `${n0(F.n)} Reuters newswires as ${M.dims} columns, with `
    + `${M.queries} held out as queries and the category of each document available for `
    + `scoring but never for searching. The metadata used for filtering is real rather than `
    + `invented, which matters more than it sounds: the rows that pass a real condition are `
    + `not scattered through the collection, they are clustered, and that is precisely what `
    + `makes filtering hard.`);

  set('filter-intro',
    `"The ten nearest documents from last month" is not a nearest neighbour query. It is a `
    + `nearest neighbour query inside a subset, and the subset is chosen by something the `
    + `index knows nothing about. There are two obvious ways to do it and they fail in `
    + `opposite directions.`);

  set('filter-note',
    `Search first and drop what fails the filter, and at ${pct(loose.selectivity)} `
    + `selectivity the recall inside the filter is ${f4(loose.post_recall)} while `
    + `${pct(loose.post_empty)} of queries come back completely empty. Filter first and scan `
    + `what is left, and the answer is exact but costs ${Math.round(tight.pre_cost)} distances `
    + `at ${pct(tight.selectivity)} instead of the ${Math.round(tight.post_cost)} the index `
    + `pays. The obvious third way, teaching the walk to skip rows that fail the filter, has `
    + `a mechanical problem that the second view measures: a search on a graph is a `
    + `<span class="bold">walk</span>, and a walk cannot cross into a component it is not in. `
    + `The kept rows form an induced subgraph, and `
    + (split
      ? `it stops being one piece as soon as the filter keeps ${pct(split.selectivity)}: at `
        + `${pct(tight.selectivity)} it is in ${tight.pieces} pieces with only `
        + `${pct(tight.biggest)} of the kept rows in the largest. The share is the number to `
        + `watch and the count of pieces is not, which the sweep makes plain: the count peaks `
        + `at ${peak.pieces} pieces when the filter keeps ${pct(peak.selectivity)} and then `
        + `falls to ${tight.pieces}, for the unremarkable reason that ${n0(tight.kept)} rows `
        + `cannot be broken into more than ${n0(tight.kept)} pieces. Fragmentation keeps `
        + `getting worse the whole way down; only its most obvious symptom turns around.`
      : `here it stays in one piece at every selectivity tested, which is the case where that `
        + `third way works and is worth knowing as much as the failure would be.`)
    + ` And with the real filter rather than a random one, the worst category leaves the `
    + `subgraph in ${worstCat.pieces} pieces at ${pct(worstCat.selectivity)} selectivity.`);

  set('churn-intro',
    `The second assumption is that the collection sits still. A graph index cannot really `
    + `delete a node, because that node's edges are part of what holds its neighbours `
    + `together, so what every implementation does is mark it and skip it during the search.`);

  set('churn-note',
    `The cost of that is visible rather than theoretical: with nothing deleted the recall `
    + `among the live rows is ${f4(churn[0].tombstone_recall)}, and with `
    + `${pct(deep.deleted)} deleted and marked it is ${f4(deep.tombstone_recall)}. `
    + (pair
      ? `Rebuilding the index from what is left recovers it to ${f4(pair.rebuilt_recall)} at `
        + `${pct(pair.deleted)} deleted, against ${f4(pair.tombstone_recall)} for the marked `
        + `one. `
      : '')
    + `The reason a rebuild is not simply the answer is arithmetic from the previous page: `
    + `this index cost ${n0(churn[0].build_cost)} distance evaluations to build, which is `
    + `${(churn[0].build_cost / churn[0].alive).toFixed(0)} queries' worth of exhaustive `
    + `search over the whole collection, and it must be paid again every time.`);

  set('hybrid-intro',
    `The third assumption is the biggest. An index returns the nearest vectors; a search `
    + `system is supposed to return the best documents, and those are not the same thing. `
    + `The oldest method in the field, counting which words a document and a query share, `
    + `has a property no embedding has: an exact match on a rare word is decisive, and a `
    + `vector that is merely close to the right direction is not.`);

  set('hybrid-note',
    `Scored on whether the top ten are about the same thing as the query, discounted by `
    + `position: at ${short.length} query words, word matching gets ${f4(short.sparse)} and `
    + `the vectors ${f4(short.dense)}; at ${long.length}, ${f4(long.sparse)} and `
    + `${f4(long.dense)}. `
    + (flips
      ? `<span class="bold">The winner changes with the query length</span>, which is why `
        + `these are two rows: a single averaged number would have been an artefact of the `
        + `length somebody picked.`
      : `${long.dense > long.sparse ? 'The vectors are ahead at both lengths' : 'Word matching is ahead at both lengths'}. `)
    + ` Fusing them by rank position alone reaches ${f4(long.fused)} and beats both of its `
    + `own inputs on ${pct(long.fused_beats_both)} of individual queries at the longer `
    + `length. Rank fusion is the crudest possible combination and that is exactly why it `
    + `works here: a term weight and a cosine similarity have no common scale, and anything `
    + `that needed one would need calibrating per corpus.`);

  set('bytes-intro',
    `Which leaves the bill. A vector database stores millions of vectors and the storage is `
    + `usually the binding constraint, so the last question is how few bytes each one can be `
    + `and what that costs.`);

  set('bytes-note',
    `From ${Math.round(float.bytes)} bytes down to ${Math.round(smallest.bytes)}, a factor of `
    + `${(float.bytes / smallest.bytes).toFixed(0)}, at a recall of `
    + `${f4(smallest.recall)}. Most of that loss comes back with the trick every serving `
    + `system uses and no comparison table mentions: retrieve a wide list with the cheap `
    + `representation, then recompute exact distances on that list alone. With `
    + `${enough.width} candidates rescored the recall is ${f4(enough.recall)}, and the `
    + `rescoring costs ${enough.width} exact distances, `
    + `${(enough.width / B.n * 100).toFixed(2)}% of the collection.`);

  set('verdict-intro',
    `Four assumptions, and the pattern is the one this whole module keeps finding: none of `
    + `them fails loudly. Each returns an answer, the answer looks fine, and the thing that `
    + `has quietly changed is what question was asked.`);

  set('verdict-filter',
    `${f4(loose.post_recall)} recall at ${pct(loose.selectivity)} selectivity and `
    + `${f4(tight.post_recall)} at ${pct(tight.selectivity)}, with `
    + `${pct(tight.post_empty)} of queries returning nothing at all.`);
  set('verdict-filter-fix',
    `Pick by selectivity, not by principle: scanning the kept rows costs `
    + `${Math.round(tight.pre_cost)} distances at ${pct(tight.selectivity)} and is exact. `
    + `Below some share of the collection the index is worse than not having one.`);
  set('verdict-churn',
    `${f4(deep.tombstone_recall)} after ${pct(deep.deleted)} of the collection is deleted and `
    + `marked, against ${f4(churn[0].tombstone_recall)} with none.`);
  set('verdict-churn-fix',
    `Rebuild on a schedule and know what it costs: ${n0(churn[0].build_cost)} distances, `
    + `which is ${(churn[0].build_cost / churn[0].alive).toFixed(0)} queries of exhaustive `
    + `search.`);
  set('verdict-hybrid',
    `Vectors alone reach ${f4(long.dense)} at ${long.length} query words and word matching `
    + `${f4(long.sparse)}.`);
  set('verdict-hybrid-fix',
    `Run both and fuse by rank: ${f4(long.fused)}, beating both inputs on `
    + `${pct(long.fused_beats_both)} of queries, with no calibration between two `
    + `incomparable scores.`);
  set('verdict-bytes',
    `${Math.round(smallest.bytes)} bytes per vector instead of ${Math.round(float.bytes)} `
    + `costs ${f4(float.recall - smallest.recall)} of recall.`);
  set('verdict-bytes-fix',
    `Rescore a shortlist exactly: ${enough.width} candidates brings it back to `
    + `${f4(enough.recall)} for ${(enough.width / B.n * 100).toFixed(2)}% of the collection `
    + `in exact distances.`);

  set('closing-1',
    `This module opened with a surface where four search methods finished within a few `
    + `thousandths of each other, and it closes with four assumptions that each cost more `
    + `than any of them. That is the thread: the engineering decisions that get argued about `
    + `are usually smaller than the ones that get made silently, and the way to tell which `
    + `is which is to measure both in the same units. A held out score, a metric, a `
    + `threshold, a split, a filter: none of these is the interesting part of a system, and `
    + `all of them decide what the interesting part is allowed to be.`);

  set('closing-2',
    `That closes the eighth branch, and with it the map. Every chip is lit and every `
    + `article links to the one after it, so the atlas now reads end to end from ordinary `
    + `least squares to a vector store. This last module is the only one that could not have `
    + `been written first: almost everything it measures is measured on data, models and `
    + `published numbers from the other seven, which is what a toolkit is for. `
    + `<a href="../">The front page</a> has the whole thing.`);

  set('ref-note',
    `Everything on this page comes from `
    + `<span class="mono">src/<wbr>utils/<wbr>generate_<wbr>vectordb_<wbr>data.py</span>, with the indexes in `
    + `<span class="mono">src/<wbr>utils/<wbr>annkit.py</span>, seed ${M.seed}. The collection is `
    + `${n0(M.index_n)} of the ${n0(M.docs)} single label Reuters newswires, in `
    + `${M.classes.length} categories over a vocabulary of ${n0(M.vocab)} words, as `
    + `${M.dims} columns; ${M.queries} queries, ${M.k} results asked for, and a graph with `
    + `${M.M} neighbours per node searched at width ${M.ef}. The `
    + `${n0(M.later)} documents held back are what the churn section inserts, so the `
    + `"arriving later" rows are real newswires and not resampled copies of the indexed ones. `
    + `Filters are applied at fixed selectivities and also by the documents' own `
    + `categories, and the connected pieces of each filtered subgraph are counted by a depth `
    + `first sweep. The word matching side is BM25 with the usual constants over the same `
    + `counts the <a href="../tf-idf/">TF-IDF article</a> measured, and fusion is reciprocal `
    + `rank with k = ${H.rrf_k}. Quality is discounted cumulative gain at ten against the `
    + `category, which is an answer key the search never sees. Cost is counted in distance `
    + `evaluations throughout, never in seconds.`);
}
