/* Every number on this page, composed from the file it was measured into. */
const IDS = ['intro-1', 'intro-2', 'cold-intro', 'cold-note', 'ceil-intro', 'ceil-math',
  'ceil-note', 'fm-intro', 'fm-math', 'tower-intro', 'cost-intro', 'cost-note',
  'verdict-intro', 'v-tower', 'v-tower-warn', 'v-cold', 'v-cold-warn', 'v-fm', 'v-fm-warn',
  'v-cross', 'v-cross-warn', 'closing-1', 'closing-2', 'closing-3', 'ref-note'];

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

const n0 = (v) => Number(v).toLocaleString('en-US');
const pc = (v, d = 1) => `${(100 * v).toFixed(d)}%`;

export function initProse(data) {
  const M = data.meta;
  const F = M.features;
  const A = data.arms;
  const C = data.ceiling;
  const P = data.pairs;
  const prev = data.previous;
  const cost = data.cost;

  const cold = A.arms;
  const contentGain = cold.content.cold.hits['10'] / Math.max(A.chance_at_10, 1e-9);
  const idsAtChance = cold.ids.cold.hits['10'] <= A.chance_at_10 * 1.5;
  const atD = C.rows.find((d) => d.k === C.d);
  const last = C.rows[C.rows.length - 1];
  const pureAtD = C.pure.find((d) => d.k === C.d);
  const here = cost.rows.find((d) => d.n >= cost.catalogue) || cost.rows[cost.rows.length - 1];

  set('intro-1',
    `The two articles before this one had one kind of input: an id. A reader was a row number, a `
    + `book was a column number, and everything either of them meant was learned from the table. `
    + `That is enough to beat the popularity baseline by a factor of `
    + `${(prev.item_knn_hits10 / prev.popular_hits10).toFixed(1)}, and it has a hole in it that no `
    + `amount of tuning closes: a book with no ratings has no column, so it has nothing to learn `
    + `from, so it is never recommended, so it never gets ratings.`);

  set('intro-2',
    `The file has always had more than ids. Every book here comes with an author, a year and a `
    + `title, which is ${n0(F.features)} features once the ${n0(F.tokens)} commonest title words, `
    + `the ${n0(F.authors)} commonest authors and ${F.decades} decades are counted, and about `
    + `${F.per_book.toFixed(1)} of them are on per book. This article is about what happens when the `
    + `model is allowed to read them, and about the shape it has to have in order to be deployed at `
    + `all.`);

  set('cold-intro',
    `The experiment first, because it is the one neither previous article could run. Take `
    + `<span class="bold">${A.cold_books}</span> books that are perfectly ordinary, with at least `
    + `fifty ratings each, and delete <span class="bold">every one of their `
    + `${n0(A.cold_cells)} training ratings</span>. Then ask each model to rank those `
    + `${A.cold_books} for the ${n0(A.cold_users)} readers who, in the held out data, went on to `
    + `give one of them four or five stars. Three towers, identical except in what the item side is `
    + `allowed to look at.`);

  set('cold-note',
    `${idsAtChance
      ? `The ids only tower scores ${pc(cold.ids.cold.hits['10'])} against a chance rate of `
        + `${pc(A.chance_at_10)}, which is the result and not a disappointment: an embedding for a `
        + `book that never appeared in a gradient is the number the initialiser wrote there.`
      : `The ids only tower manages ${pc(cold.ids.cold.hits['10'])} against a chance rate of `
        + `${pc(A.chance_at_10)}, above chance, which needs explaining rather than explaining away, `
        + `because an embedding that never took a gradient cannot know anything about the book it `
        + `belongs to. The explanation is the grey bar: recommending the most rated of the three `
        + `hundred scores ${pc(A.arms.popular.cold.hits['10'])}, and what the tower carries besides `
        + `its embeddings is a popularity term per item, which the sampling trains for every book `
        + `whether or not anybody rated it.`} `
    + `${contentGain >= 1.35
      ? `Reading the features instead scores ${pc(cold.content.cold.hits['10'])}, `
        + `<span class="bold">${contentGain.toFixed(1)} times chance</span>, from an author, a `
        + `decade and a handful of title words, and carrying both `
        + `${pc(cold.both.cold.hits['10'])}. `
        + `${A.arms.popular.cold.hits['10'] > cold.both.cold.hits['10']
          ? `And then the grey bar, which is the reason the first article of this module insisted `
            + `on running it: recommending the most rated of the three hundred, with no model and `
            + `no features, scores <span class="bold">${pc(A.arms.popular.cold.hits['10'])}</span> `
            + `and beats all three. These books were chosen with at least fifty ratings each, so `
            + `"cold" here means the model was not shown them, not that nobody has read them, and `
            + `popularity survives that. The honest reading is the one the bars give: features `
            + `recover most of what deleting the history destroyed, and they do not recover the `
            + `part that was popularity.`
          : `And the popularity control, which the first article of this module insisted on, does `
            + `${pc(A.arms.popular.cold.hits['10'])}: the features beat it.`}`
      : `Reading the features instead scores ${pc(cold.content.cold.hits['10'])}, which is `
        + `${contentGain.toFixed(2)} times chance and therefore <span class="bold">not a `
        + `result</span>. At this budget, an author, a decade and a handful of title words did not `
        + `buy a usable cold start on this catalogue, and the honest thing is to say so: the `
        + `mechanism is sound and the evidence for it here is not. What the same experiment does `
        + `establish is the negative half, which needs no budget at all, because it is structural: `
        + `an id embedding for a book with no ratings cannot be anything but noise.`} `
    + `And the warm view is the other half of the sentence: on the ordinary catalogue the same `
    + `content only tower scores ${pc(cold.content.warm.hits['10'])} against `
    + `${pc(cold.ids.warm.hits['10'])} for ids alone. `
    + `${cold.ids.warm.hits['10'] > cold.content.warm.hits['10']
      ? 'So features are what you have when history is missing, and history is better than '
        + 'features when you have it, which is the trade the third arm exists to avoid making.'
      : 'So even with the history there, the features are not the weaker half at this budget, '
        + 'which is not the textbook order and is what the numbers say: six passes is not enough '
        + 'to learn ten thousand independent embeddings, and it is plenty to learn a few thousand '
        + 'shared ones.'}`
    + `<br /><br />`
    + `${data.content_knn
      ? `Those four bars all depend on how well three towers trained, which is a bad thing for an `
        + `argument to depend on, so here is the same question asked without training anything at `
        + `all. Score each of the ${A.cold_books} deleted books by the cosine between its features `
        + `and the reader's own profile, which is the sum of the features of everything they liked, `
        + `weighted by rarity. No parameters, no learning rate, no budget: `
        + `<span class="bold">${pc(data.content_knn.cold.hits['10'])}</span> against `
        + `${pc(A.chance_at_10)} for chance and ${pc(A.arms.popular.cold.hits['10'])} for `
        + `popularity. `
        + `${data.content_knn.cold.hits['10'] > Math.max(A.chance_at_10, A.arms.popular.cold.hits['10'])
          ? 'That is the result the section was after, and it is the one that does not move if you '
            + 'retrain anything: an author and a handful of title words are worth something for a '
            + 'book with no readers.'
          : 'Which does not clear the two controls, and is therefore the honest answer for this '
            + 'catalogue: the features here are too thin to place a book nobody has read.'}`
      : ''}`);

  set('ceil-intro',
    `Now the constraint that the name hides. Two towers means the score of a pair is `
    + `\\(s(u, i) = f(u)^{\\top} g(i)\\): the reader goes through one network, the item through `
    + `another, and they meet only at the very end, in a dot product. Write out the whole matrix of `
    + `scores that such a model can produce and it is a product of an \\(n \\times k\\) by a `
    + `\\(k \\times m\\), which is a matrix of <span class="bold">rank at most k</span>. That is not `
    + `a bound on how well it trains. It is a bound on what it can say.`);

  set('ceil-math',
    `And it is exactly computable, because <a href="../matrix-factorization/">the previous `
    + `article</a> already met the theorem that does it: the best rank k approximation of any matrix `
    + `in squared error is its truncated SVD. So in a world where the true score of every pair is `
    + `known, the best possible two tower model of dimension k is a number you can look up rather `
    + `than a result you have to train for. `
    + `$$\\min_{\\operatorname{rank}(S) \\le k} \\|S^{\\star} - S\\|_F^2 = \\sum_{j > k} \\sigma_j^2$$ `
    + `The world below has ${C.d} tastes, which a dot product can represent exactly, plus one cross `
    + `term worth ${pc(C.cross_share)} of the signal, which it cannot.`);

  set('ceil-note',
    `The two views are the whole argument. With the cross term in, the matrix has rank `
    + `${C.rank_needed} and the error at ${C.d} numbers a side is ${atD.rmse.toFixed(4)}, still `
    + `${last.rmse.toFixed(4)} at ${last.k}. With it out, the same curve hits `
    + `${pureAtD.rmse.toExponential(1)} at exactly ${C.d}. A cross term is any effect that depends `
    + `on the pair and cannot be written as something about the reader times something about the `
    + `item: "this reader likes long books by this particular author", "this reader reads horror `
    + `only in October". Those are the effects a ranking model is added later to catch, and the `
    + `reason the industry standard is two stages rather than one.`);

  set('fm-intro',
    `Which raises the question of how a model represents a cross at all. The honest version is a `
    + `weight per pair of features, and the counting says why nobody does that: with `
    + `${n0(P.features)} features there are ${n0(P.pairs)} pairs, and over a sample of readers only `
    + `<span class="bold">${pc(P.share, 2)}</span> of those pairs ever occur together at all. A `
    + `weight fitted on a pair that never occurs is not a small weight; it is an unfitted one.`);

  set('fm-math',
    `A factorisation machine writes the weight of a pair as a dot product of two vectors, one per `
    + `feature, `
    + `$$\\hat{y}(x) = w_0 + \\sum_i w_i x_i + \\sum_{i < j} \\langle v_i, v_j \\rangle x_i x_j$$ `
    + `which turns ${n0(P.params_full)} free numbers into ${n0(P.params_fm)}, a factor of `
    + `${P.ratio.toFixed(0)}, and does something the count does not show: a pair that never occurs `
    + `still gets a weight, because both of its features occurred with other things. That is the `
    + `same trick as the previous article, applied one level up. The tower on this page is the same `
    + `idea again: an item's vector is the sum of its feature vectors, so two books that share an `
    + `author start out close without anybody saying so. DeepFM adds a small network over the same `
    + `embeddings, keeping the factorised second order term rather than hoping the network `
    + `rediscovers it.`);

  set('tower-intro',
    `Here is the retrieval half, running in the page. The item vectors were computed once by the `
    + `generator and frozen; picking books builds a query vector; scoring the catalogue is one dot `
    + `product per item. That is the entire deployment story, and it is why the ceiling above is `
    + `accepted rather than fixed.`);

  set('cost-intro',
    `The arithmetic is worth counting rather than describing, because it is what decides the `
    + `architecture. Nothing here is a wall clock: these are multiplications, so they are the same `
    + `on any machine.`);

  set('cost-note',
    `At ${n0(cost.catalogue)} items the difference is a factor of ${here.ratio.toFixed(0)}; at a `
    + `million it is ${cost.rows[cost.rows.length - 1].ratio.toFixed(0)}. And that comparison is `
    + `generous to the pair model, because it counts a full scan on both sides. The two tower `
    + `vectors do not depend on the query, so in practice they go into an index that answers "the `
    + `nearest few thousand" without looking at the rest, and the pair model has no such option: `
    + `its input does not exist until the pair does. This is why production systems retrieve with `
    + `one and rank with the other, and why the ceiling this article measures is a design decision `
    + `rather than an oversight.`);

  set('verdict-intro',
    `Four choices, and the first two are about deployment rather than accuracy.`);

  set('v-tower', `One dot product per item, and item vectors that can be computed in advance: `
    + `${n0(here.tower)} multiplications against ${n0(here.cross)} for a model that sees the pair.`);
  set('v-tower-warn', `The score matrix it can write has rank ${M.k}. Anything that needs the pair `
    + `is outside what it can express, measurably.`);

  set('v-cold', `${contentGain >= 1.35
    ? `${pc(cold.content.cold.hits['10'])} on books with no training ratings at all, against `
      + `${pc(A.chance_at_10)} for chance and ${pc(cold.ids.cold.hits['10'])} for ids.`
    : `Structurally, the only arm that can say anything at all: ids score `
      + `${pc(cold.ids.cold.hits['10'])} against ${pc(A.chance_at_10)} for chance, and content `
      + `${pc(cold.content.cold.hits['10'])}, which at this budget is the same thing.`}`);
  /* this row used to assert the textbook order, that features are the weaker
     signal once history exists, while printing the two numbers that say the
     opposite. Both halves are composed now. */
  set('v-cold-warn', `${cold.ids.warm.hits['10'] > cold.content.warm.hits['10']
    ? `On warm items the features are the weaker signal `
      + `(${pc(cold.content.warm.hits['10'])} against ${pc(cold.ids.warm.hits['10'])}), so the `
      + `useful arm is the one that carries both.`
    : `The textbook order did not appear here: on warm items the same features still score `
      + `${pc(cold.content.warm.hits['10'])} against ${pc(cold.ids.warm.hits['10'])} for ids, `
      + `which is a statement about this budget rather than about the method, and the arm to `
      + `deploy is still the one that carries both.`}`);

  set('v-fm', `${n0(P.params_fm)} numbers instead of ${n0(P.params_full)}, and a weight for pairs `
    + `that never occur together, which is ${pc(1 - P.share, 2)} of them.`);
  set('v-fm-warn', `It buys sharing, not magic: a factorised weight is still a rank constraint on `
    + `the interaction matrix.`);

  set('v-cross', `Whatever is left above the rank ceiling: ${atD.rmse.toFixed(4)} of error that `
    + `${C.d} numbers a side cannot remove in the world measured here.`);
  set('v-cross-warn', `You pay for it per item scored, which is why it runs on the few hundred the `
    + `first stage returned and not on the catalogue.`);

  set('closing-1',
    `The through line of this branch has been one table getting more and more structure. Compare `
    + `columns of it, and you get neighbours. Factorise it, and you get a vector per row and column. `
    + `Let those vectors be computed from features instead of looked up, and a book nobody has read `
    + `still has one. Each step buys something and each one is a constraint: the last two are both `
    + `rank constraints, and the rank is the thing you can compute the cost of exactly.`);

  set('closing-2',
    `What none of the three can do is say anything about a relationship that is not "these two `
    + `co-occurred". A reader and a book, in this module, are two ends of one kind of edge. But the `
    + `objects people actually want to reason about have many kinds: this author wrote that book, `
    + `which is in this series, which belongs to that genre, which this reader follows.`);

  set('closing-3',
    `That is a graph, and the second half of this module is about learning on one. The first `
    + `question is the one this article's cold start experiment asked in miniature: if a node's `
    + `description can come from its neighbours instead of from itself, what does that buy, and `
    + `when does it cost? <a href="../gnn/">The average of your neighbours</a> is next.`);

  set('ref-note',
    `The ratings, the split and the held out cells are the two previous articles', asserted by `
    + `fingerprint (<code>${M.split_fingerprint}</code>) rather than declared. The towers are `
    + `${M.k} numbers a side, trained with a sampled softmax over ${M.neg} shared negatives per `
    + `batch of ${M.batch} for ${M.epochs} passes, on the ratings of four and five stars. The `
    + `features are ${n0(F.tokens)} title words, ${n0(F.authors)} authors and ${F.decades} decades; `
    + `${n0(F.unknown_author_books)} books fall outside the author list and get a shared token for `
    + `it. The cold start arm deletes ${n0(A.cold_cells)} training cells and is scored on the `
    + `${A.cold_books} books they belonged to, ranked among themselves, so chance is `
    + `${pc(A.chance_at_10)} rather than one in ten thousand. The ceiling section is a world of `
    + `${C.n_users} by ${C.n_items} whose true scores are written by the generator, so its best `
    + `possible error is a singular value sum and not a training outcome. From `
    + `<code>src/utils/generate_tower_data.py</code>.`);

  return IDS;
}
