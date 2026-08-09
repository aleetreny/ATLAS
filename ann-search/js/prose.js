/* Every number in the running text, composed from the file at load. */
const n0 = (v) => Number(v).toLocaleString('en-US');
const f3 = (v) => Number(v).toFixed(3);
const f4 = (v) => Number(v).toFixed(4);

export const PROSE_IDS = [
  'opening-note', 'conc-intro', 'conc-note', 'trade-intro', 'trade-note',
  'demo-intro', 'abl-intro', 'abl-note', 'verdict-intro',
  'verdict-brute', 'verdict-brute-warn', 'verdict-ivf', 'verdict-ivf-warn',
  'verdict-hnsw', 'verdict-hnsw-warn', 'verdict-pq', 'verdict-pq-warn',
  'closing-1', 'closing-2', 'ref-note',
];

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

export function initProse(data) {
  const M = data.meta;
  const C = data.concentration;
  const I = data.indices;
  const A = data.ablation;
  const S = data.scale;

  const target = 0.9;
  const cheapest = (rows) => rows.slice().sort((a, b) => a.cost - b.cost)
    .find((r) => r.recall >= target);
  const hn = cheapest(I.hnsw);
  const iv = cheapest(I.ivf);
  const pq = cheapest(I.pq);
  const bestPq = I.pq.reduce((a, b) => (b.recall > a.recall ? b : a));
  const smallPq = I.pq.reduce((a, b) => (b.bytes < a.bytes ? b : a));
  const first = S[0];
  const last = S[S.length - 1];
  const growth = Math.log(last.cost / first.cost) / Math.log(last.n / first.n);
  const efs = Array.from(new Set(A.rows.map((r) => r.ef))).sort((a, b) => a - b);
  const mid = efs[Math.floor(efs.length / 2)];
  const at = (arm) => A.rows.find((r) => r.arm === arm && r.ef === mid);
  const layerCost = at('full').recall - at('one layer').recall;
  const pruneCost = at('full').recall - at('nearest m edges').recall;

  set('opening-note',
    `The collection is ${n0(I.n)} of the Reuters newswires this atlas has used since the `
    + `<a href="../tf-idf/">TF-IDF article</a>, as ${M.dims} dense columns, with `
    + `${I.queries} held out as queries. Exhaustive search over them costs `
    + `${Math.round(I.brute_cost)} distances per query and is the answer key: every recall on `
    + `this page is measured against it, not against another index. And because those `
    + `newswires carry a category, there is a second answer key that matters more, which the `
    + `last view of the second widget uses.`);

  set('conc-intro',
    `Before asking how to find the nearest neighbour quickly, it is worth asking whether the `
    + `nearest neighbour is a distinguishable object. In enough dimensions of independent `
    + `noise it stops being one: every point is about as far away as every other, so the `
    + `nearest and the farthest converge and an approximate answer is indistinguishable from `
    + `an exact one for the wrong reason.`);

  set('conc-note',
    `At ${C.gaussian[C.gaussian.length - 1].d} dimensions of noise the farthest of two `
    + `thousand points is only ${C.gaussian[C.gaussian.length - 1].ratio.toFixed(2)} times as `
    + `far as the nearest. Real vectors do not behave that way at the same column count, and `
    + `the reason is measurable rather than hopeful: the newswire vectors have ${C.reuters.d} `
    + `columns and an intrinsic dimension of `
    + `<span class="bold">${C.reuters.intrinsic.toFixed(1)}</span>, estimated from the ratio `
    + `between each point's first and second neighbour, so their contrast is `
    + `${f3(C.reuters.contrast)} where noise of the same width gives `
    + `${f3(C.gaussian.find((r) => r.d === C.reuters.d).contrast)}. Approximate search works `
    + `on real data because real data is thinner than the box it is written in. When it stops `
    + `working, that is usually the first thing to measure.`);

  set('trade-intro',
    `Three ways to avoid looking at everything, each with one dial. The inverted file splits `
    + `the collection into cells with k-means, which is the algorithm the `
    + `<a href="../kmeans/">clustering article</a> spent its length on, and looks only in the `
    + `nearest few. The navigable graph connects each vector to a few neighbours and walks `
    + `greedily downhill. The product quantiser does not skip any vector at all: it makes `
    + `each one cheaper by splitting it into chunks and replacing each chunk with its nearest `
    + `entry in a small codebook, which is the same object the `
    + `<a href="../vq-vae/">discrete autoencoder article</a> built.`);

  set('trade-note',
    `To reach ${(target * 100).toFixed(0)}% of the true neighbours, the graph needs `
    + `${hn ? Math.round(hn.cost) : 'more than its widest setting allows'} distances, the `
    + `inverted file ${iv ? Math.round(iv.cost) : 'more than its widest setting allows'}, `
    + `against ${Math.round(I.brute_cost)} for looking at everything: `
    + (hn ? `a factor of ${(I.brute_cost / hn.cost).toFixed(1)} for the graph` : 'no saving')
    + `. Two things about that number are worth saying. It only counts distances, so it `
    + `ignores that a graph walk jumps around memory while a scan does not, which is why the `
    + `crossover in seconds is at a larger collection than the crossover in distances. And it `
    + `grows slowly: from ${n0(first.n)} to ${n0(last.n)} vectors the graph's cost per query `
    + `goes from ${Math.round(first.cost)} to ${Math.round(last.cost)} distances while `
    + `exhaustive search goes from ${n0(first.brute)} to ${n0(last.brute)}, so the saving `
    + `widens from ${first.ratio.toFixed(1)} times to ${last.ratio.toFixed(1)}. That is `
    + `the ${growth.toFixed(2)} power of the collection size rather than the first power `
    + `exhaustive search pays. At its widest setting the inverted file reaches ${I.ivf[I.ivf.length - 1].recall.toFixed(4)} recall `
    + `after ${Math.round(I.ivf[I.ivf.length - 1].cost)} distance evaluations, slightly more `
    + `than exhaustive search: a wider dial is not automatically a better setting. The second view is the one that changes what all of this is `
    + `worth, and it is in the readout.`);

  set('demo-intro',
    `The graph is worth seeing rather than describing, so here is one: ${data.demo.points.length} `
    + `of the same vectors projected onto two coordinates and indexed in that plane, so the `
    + `picture is a whole index and not a shadow of one. The search enters at the sparsest `
    + `layer, walks greedily to the closest node it can reach, drops a layer, and repeats.`);

  set('abl-intro',
    `The graph index has two pieces and they get very different amounts of attention. The `
    + `layers are what the name is about. The edge pruning is a rule that decides which `
    + `neighbours a node keeps: not simply the closest ones, but only those that are closer `
    + `to the node than to a neighbour already kept, which stops every edge pointing the same `
    + `way into a dense region.`);

  set('abl-note',
    `Removing them one at a time, at the middle search width: the layers are worth `
    + `${f4(layerCost)} of recall and the pruning is worth ${f4(pruneCost)}. `
    + (Math.abs(pruneCost) > Math.abs(layerCost)
      ? `<span class="bold">The piece nobody names is doing more than the piece the algorithm `
        + `is named after</span>, which is what later work on this family found: the `
        + `hierarchy earns its keep when the intrinsic dimension is high, and `
        + `${C.reuters.intrinsic.toFixed(1)} is not high.`
      : `The hierarchy is doing more, which is the ordering the original paper claims.`)
    + ` And the column that decides whether any of it is worth building: the full index pays `
    + `${n0(at('full').build)} distances before it answers a single query, which is `
    + `${(at('full').build / I.brute_cost).toFixed(0)} queries' worth of exhaustive search.`);

  set('verdict-intro',
    `Four rows on the same axis. The one thing they share is that none of them is faster in `
    + `any absolute sense: each buys a lower cost per query with a cost somewhere else, and `
    + `the somewhere else is the column on the right.`);

  set('verdict-brute',
    `${Math.round(I.brute_cost)} distances per query, ${I.float_bytes} bytes per vector, and `
    + `the right answer every time.`);
  set('verdict-brute-warn',
    `Nothing, which is why it is the answer key. It is also the only one here that needs no `
    + `build and handles a changing collection for free.`);
  set('verdict-ivf',
    iv ? `${Math.round(iv.cost)} distances for ${(target * 100).toFixed(0)}% recall, a factor `
      + `of ${(I.brute_cost / iv.cost).toFixed(1)}.`
      : `never reaches ${(target * 100).toFixed(0)}% recall within its dial range here.`);
  set('verdict-ivf-warn',
    `A vector whose true neighbours are in a cell it did not probe is simply not found, and `
    + `no widening of the search inside the probed cells will recover it.`);
  set('verdict-hnsw',
    hn ? `${Math.round(hn.cost)} distances for ${(target * 100).toFixed(0)}% recall, a factor `
      + `of ${(I.brute_cost / hn.cost).toFixed(1)}, with cost growing as the `
      + `${growth.toFixed(2)} power of the collection.`
      : `does not reach ${(target * 100).toFixed(0)}% recall within its dial range here.`);
  set('verdict-hnsw-warn',
    `${n0(I.build_cost)} distances to build and a graph to keep in memory: `
    + (I.degrees.min === I.degrees.max
      ? `<span class="bold">exactly ${I.degrees.max}</span> edges on every node at the bottom `
        + `layer, not an average of that, because the cap is what every node reaches`
      : `${(I.degrees.mean).toFixed(1)} edges per node at the bottom layer on average, from `
        + `${I.degrees.min} to ${I.degrees.max}`)
    + `, on top of the vectors themselves.`);
  set('verdict-pq',
    `${smallPq.bytes} bytes per vector instead of ${I.float_bytes}, a factor of `
    + `${(I.float_bytes / smallPq.bytes).toFixed(0)}, at ${f3(smallPq.recall)} recall; with `
    + `${bestPq.bytes} bytes it reaches ${f3(bestPq.recall)}.`);
  set('verdict-pq-warn',
    `It still touches every vector, so it saves memory and arithmetic rather than visits. `
    + `Its distances are estimates: the error on them is ${f3(bestPq.adc_err)} relative at `
    + `its most accurate setting.`);

  set('closing-1',
    `The shape of this page is one trade drawn four times, and the honest summary of it is `
    + `the second view of the second widget. Every index here loses some true neighbours, and `
    + `the loss looks bad on a recall axis and much smaller on the axis anybody actually `
    + `cares about, because the neighbours that are easy to find are the ones that agree with `
    + `the query. That is the argument for approximate search, and it is not an argument `
    + `about speed.`);

  set('closing-2',
    `What none of it survives contact with is a collection that changes, or a query that `
    + `carries a condition, or an application that wants the best documents rather than the `
    + `nearest vectors. <a href="../vector-db/">The last article of this module</a> is about `
    + `those three, and each of them breaks something this page took for granted.`);

  set('ref-note',
    `Everything on this page comes from `
    + `<span class="mono">src/<wbr>utils/<wbr>generate_<wbr>annsearch_<wbr>data.py</span> and the indexes in `
    + `<span class="mono">src/<wbr>utils/<wbr>annkit.py</span>, seed ${M.seed}. The collection is `
    + `${n0(M.index_n)} of the ${n0(M.docs)} single label Reuters newswires, in `
    + `${M.classes.length} categories over a vocabulary of ${n0(M.vocab)} words, as `
    + `${M.dims} columns from a singular value `
    + `decomposition of their weighted counts, with ${I.queries} held out as queries and `
    + `${M.k} neighbours asked for. The graph is built with M = ${M.M} neighbours per node, `
    + `and its bottom layer reaches the 2M cap of ${I.degrees.max}; construction uses a width `
    + `of ${M.ef_construction}. The inverted file uses ${M.nlist} cells `
    + `from k-means; the product quantiser uses eight bit codebooks. Cost is counted inside `
    + `the indexes as distance evaluations and never in seconds, which is the only form that `
    + `means the same thing on another machine. The intrinsic dimension is the two nearest `
    + `neighbour estimator of Facco and colleagues, fitted through the origin, over a sample `
    + `of ${n0(C.reuters.dim_sample)} points with exact duplicates excluded and counted. The `
    + `word vectors are read from the <a href="../embeddings/">embeddings article</a>'s own `
    + `file, ${C.word_meta.shipped} words of ${C.word_meta.corpus} in `
    + `${C.word_meta.dim} dimensions, decoded from the float16 it ships them in.`);
}
