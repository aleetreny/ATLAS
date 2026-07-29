/* Every number on this page, composed from the file it was measured into. */
const IDS = ['intro-1', 'intro-2', 'intro-3', 'live-intro', 'sym-intro', 'sym-math', 'sym-note',
  'pat-intro', 'pat-note', 'proto-intro', 'proto-note', 'verdict-intro',
  'v-transe', 'v-transe-warn', 'v-rotate', 'v-rotate-warn', 'v-distmult', 'v-distmult-warn',
  'v-proto', 'v-proto-warn', 'closing-1', 'closing-1b', 'closing-2', 'closing-3',
  'ref-note'];

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

const n0 = (v) => Number(v).toLocaleString('en-US');
const pc = (v, d = 1) => `${(100 * v).toFixed(d)}%`;
const nice = (n) => n.replace(/^_/, '').replace(/_/g, ' ');

export function initProse(data) {
  const M = data.meta;
  const R = data.relations;
  const W = data.wn;
  const P = data.patterns;

  const sym = R.filter((r) => r.symmetry > 0.5);
  const anti = R.filter((r) => r.symmetry <= 0.5);
  const biggest = R[0];
  const meanNorm = (rows, model, field) => rows.reduce(
    (a, r) => a + W[model].relations[String(r.id)][field], 0) / Math.max(rows.length, 1);
  const tSym = meanNorm(sym, 'transe', 'norm');
  const tAnti = meanNorm(anti, 'transe', 'norm');
  const rSym = meanNorm(sym, 'rotate', 'self_inverse');
  const rAnti = meanNorm(anti, 'rotate', 'self_inverse');
  const best = ['transe', 'rotate', 'distmult'].reduce(
    (a, k) => (W[k].eval.mrr > W[a].eval.mrr ? k : a));
  const NAME = { transe: 'TransE', rotate: 'RotatE', distmult: 'DistMult' };
  const comp = P.composition;

  set('intro-1',
    `<a href="../gnn/">The previous article</a> spent itself on one operation: mix a row with the `
    + `rows next to it. Every edge went into the same average, which quietly says that all edges `
    + `mean the same thing. The graphs people keep in databases are not like that. "Cites", "was `
    + `written by" and "is a kind of" are three different claims about the same pair, and the third `
    + `one has a direction the other two do not.`);

  set('intro-2',
    `So the object here is a triple: a head, a relation and a tail. `
    + `<span class="bold">${n0(M.entities)}</span> entities, `
    + `<span class="bold">${M.relations}</span> relations and `
    + `<span class="bold">${n0(M.train)}</span> facts, from WN18RR, which is WordNet with the `
    + `trivially invertible relations removed so that a model cannot pass by copying the answer `
    + `from the other side. The task is completion: hide the tail of a fact, score every one of the `
    + `${n0(M.entities)} entities as a candidate, and see where the true one lands.`);

  set('intro-3',
    `What makes this section different from everything else in the module is that the interesting `
    + `question has closed form answers. A scoring function is a promise about what kinds of `
    + `relation it can represent, and for the two on this page you can prove what happens to a `
    + `symmetric relation before running anything. Then you can train the model and go and look at `
    + `the parameter the proof was about.`);

  set('live-intro',
    `Start with the picture, because it is the model. RotatE puts every entity somewhere on the `
    + `complex plane, in ${M.dim / 2} planes at once, and a relation is a rotation about the `
    + `origin: to check a fact, turn the head and see how far you land from the tail. Pick a `
    + `relation and turn it by hand; the fitted angle is the one that puts the filled points on `
    + `top of the dark ones.`);

  set('sym-intro',
    `Now the argument. TransE says a fact is true when \\(h + r \\approx t\\), a relation is a `
    + `step, and checking a fact is walking it. Suppose the relation is symmetric, so that every `
    + `fact comes with its mirror.`);

  set('sym-math',
    `$$h + r \\approx t \\quad\\text{and}\\quad t + r \\approx h \\;\\Longrightarrow\\; 2r \\approx 0 `
    + `\\;\\Longrightarrow\\; r \\approx 0$$ `
    + `and a relation vector of zero says that every pair it joins is at the same point, which is `
    + `not a description of anything. RotatE replaces the step with a turn, \\(t \\approx h \\circ `
    + `r\\) with \\(|r_i| = 1\\), and the same requirement becomes \\(r^2 = 1\\): the rotation must `
    + `be its own inverse, so each of its angles has to be a half turn or none at all. Two `
    + `different predictions about numbers inside a trained model, and this graph has `
    + `${sym.length} symmetric relations and ${anti.length} that are not, so both can be checked.`);

  set('sym-note',
    `${tSym < tAnti
      ? `TransE does exactly what the algebra says: relation vectors of average length `
        + `<span class="bold">${tSym.toFixed(3)}</span> on the symmetric relations against `
        + `<span class="bold">${tAnti.toFixed(3)}</span> on the rest. Nothing in the training `
        + `objective mentions symmetry; it is the only place the optimiser can go.`
      : `On this run the separation TransE's algebra predicts does not appear (${tSym.toFixed(3)} `
        + `against ${tAnti.toFixed(3)}), which is a fact about this optimum rather than about the `
        + `proof, and it is reported rather than tuned away.`} `
    + `${rSym < rAnti
      ? `RotatE, on the same relations, lands its symmetric rotations `
        + `<span class="bold">${rSym.toFixed(3)}</span> away from being self inverse against `
        + `<span class="bold">${rAnti.toFixed(3)}</span> for the others, so it has found the half `
        + `turns without being told they exist.`
      : `RotatE's rotations do not separate the two groups here (${rSym.toFixed(3)} against `
        + `${rAnti.toFixed(3)}), which is worth saying plainly.`} `
    + `On ${n0(M.eval_triples)} test facts scored against all ${n0(M.entities)} entities, `
    + `${NAME[best]} leads with a mean reciprocal rank of `
    + `<span class="bold">${W[best].eval.mrr.toFixed(4)}</span>, against `
    + `${['transe', 'rotate', 'distmult'].filter((k) => k !== best)
      .map((k) => `${W[k].eval.mrr.toFixed(4)} for ${NAME[k]}`).join(' and ')}. `
    /* the tidy version of this sentence is "the ranking follows the
       expressiveness", and on this run it does not: the model that cannot
       represent a direction at all finishes above the one that can. Saying so
       is the whole reason the page ranks them. */
    + `${W.distmult.eval.mrr > W.transe.eval.mrr
      ? `And that ordering is worth stopping on, because the middle of it is not where a reading `
        + `of the scoring functions would put it. DistMult scores a triple and its reverse `
        + `identically, so on a graph where <span class="bold">${pc(1 - R.filter((r) => r.symmetry > 0.5).reduce((a, r) => a + r.n, 0) / R.reduce((a, r) => a + r.n, 0), 0)}</span> `
        + `of the facts belong to a relation that is not symmetric it is representing something it `
        + `provably cannot represent, and it still finishes above TransE. Expressiveness bounds what `
        + `a model can say; it does not order what a fit of this size will find.`
      : 'And the ranking follows the expressiveness, with the least expressive of the three last.'}`);

  set('pat-intro',
    `WN18RR mixes its patterns: ${nice(biggest.name)} is the biggest relation with `
    + `${n0(biggest.n)} facts and it is hierarchical, others are symmetric, and an average over `
    + `eleven of them hides the only thing worth knowing. So the same three models are also fitted `
    + `to a graph written here, where each relation is one pattern on purpose: a symmetric one, a `
    + `pair of inverses, a composition of two steps, and a hierarchy where many entities share one `
    + `parent.`);

  /* This section was written expecting the identities to hold and reporting how
     tightly. They do not hold, and that turned out to be the better half of the
     article, so the paragraph now says why the two predictions behave
     differently instead of grading them on the same scale. */
  const compHolds = comp && comp.parent_plus_parent < 0.25;
  const invHolds = comp && comp.inverse < 0.25;
  set('pat-note',
    `${comp
      ? `The composition is worth doing as algebra rather than as a metric. If a grandparent is a `
        + `parent twice, then TransE's vectors ought to satisfy \\(r_{gp} = 2 r_{p}\\), and the `
        + `fitted ones miss that by <span class="bold">${pc(comp.parent_plus_parent)}</span> of the `
        + `vector's own length. The inverse pair ought to satisfy \\(r_{child} = -r_{parent}\\), and `
        + `they miss by <span class="bold">${pc(comp.inverse)}</span>. `
        + `${compHolds && invHolds
          ? 'Both identities are there in the parameters, on a graph where each pattern was '
            + 'written in by hand.'
          : `Neither identity is there, and that is the most useful measurement on this page, `
            + `because two paragraphs ago the other prediction landed exactly. The difference is `
            + `not how hard the two are; it is what kind of statement each one is. A symmetric `
            + `relation <span class="bold">forces</span> \\(r\\) towards zero: every step of `
            + `gradient descent on a mirrored fact pushes it there and there is nowhere else to `
            + `go, so the optimiser arrives whether or not it is trying to. Composition only `
            + `<span class="bold">permits</span> \\(r_{gp} = 2 r_{p}\\): it is one of the many `
            + `ways to fit the grandparent facts, it costs nothing extra to ignore, and a fit this `
            + `size does not go looking for it. A proof about what a model can represent is not a `
            + `promise about what a finite run will find, and the two halves of this page are what `
            + `that distinction looks like measured.`} `
        + `${comp.norm_mirror < comp.norm_parent
          ? `The symmetric relation's vector did come out `
            + `${(comp.norm_parent / Math.max(comp.norm_mirror, 1e-9)).toFixed(2)} times shorter `
            + `than the hierarchy's here, in the same direction as the real graph and nothing like `
            + `as far: ${(tAnti / Math.max(tSym, 1e-9)).toFixed(1)} times there, on relations `
            + `nobody wrote by hand.`
          : `The symmetric relation's vector did not collapse here, which is the one place this `
            + `written graph disagrees with the real one.`}`
      : 'The composition check could not be computed on this run.'}`);

  set('proto-intro',
    `One number on this page is quoted more than every other number in this literature combined, `
    + `and it is not a property of a model. Scoring a test fact means ranking the true tail against `
    + `every entity, and some of the others are also true: the graph says a dog is a kind of `
    + `mammal, and it also says a dog is a kind of animal. Counting those as mistakes measures the `
    + `graph's density rather than the model.`);

  set('proto-note',
    `Filtering them out is what everybody does and it is worth seeing the size of: for `
    + `${NAME[best]} the mean reciprocal rank is `
    + `<span class="bold">${W[best].eval.mrr.toFixed(4)}</span> filtered and `
    + `<span class="bold">${W[best].eval.mrr_raw.toFixed(4)}</span> raw, a difference of `
    + `${((W[best].eval.mrr / Math.max(W[best].eval.mrr_raw, 1e-9) - 1) * 100).toFixed(0)}%, which `
    + `is larger than most of the gaps this leaderboard is decided by. It also cuts both ways: `
    + `filtering uses the test set to score the test set, which is fine for comparing models on the `
    + `same graph and means nothing about how a model would behave on a fact nobody has recorded.`);

  set('verdict-intro',
    `Four rows, and the first three are decided by what your relations are rather than by which `
    + `paper is newest.`);

  /* parent_plus_parent is the MISS, not the fit, so "satisfy it to 12%" said
     the opposite of the number it printed */
  set('v-transe', `${compHolds
    ? `Hierarchies and chains for free: composition is addition, and on the written graph the `
      + `fitted \\(r_{gp}\\) misses \\(2 r_{p}\\) by only ${pc(comp.parent_plus_parent)}.`
    : `The cheapest model here, and the one whose failure is legible: a chain of two relations is `
      + `a sum of two vectors, so it <span class="bold">can</span> compose. On this budget it did `
      + `not (the fitted \\(r_{gp}\\) misses \\(2 r_{p}\\) by `
      + `${comp ? pc(comp.parent_plus_parent) : 'more than the identity allows'}), which is a `
      + `statement about the run and not about the family.`}`);
  set('v-transe-warn', `A symmetric relation forces its vector to zero, and on the real graph the `
    + `${tSym < tAnti
      ? `symmetric ones came out ${(tAnti / Math.max(tSym, 1e-9)).toFixed(1)} times shorter.`
      : 'symmetric ones did not come out shorter at this budget, which is reported rather than '
        + 'tuned away: the proof is about the minimum, not about where a short run stops.'}`);

  set('v-rotate', `MRR ${W.rotate.eval.mrr.toFixed(4)} on WN18RR at ${M.dim} dimensions, and `
    + `symmetry costs it nothing: a half turn is its own inverse.`);
  set('v-rotate-warn', `Twice the parameters per relation are not the price; the price is that a `
    + `rotation cannot change a length, so it cannot express a relation that should.`);

  const dMirror = P.models.distmult.per_pattern.mirror.mrr;
  const dHier = P.models.distmult.per_pattern.parent.mrr;
  set('v-distmult', `MRR ${W.distmult.eval.mrr.toFixed(4)} over all eleven relations, and `
    + `${dMirror > dHier
      ? `on a purely symmetric relation it is the right model rather than a weaker one `
        + `(${dMirror.toFixed(3)} on the written one).`
      : `${dMirror.toFixed(3)} on the written symmetric relation, which is the column its `
        + 'scoring function was supposed to own.'}`);
  set('v-distmult-warn', `It scores a triple and its reverse identically, so on the hierarchy it `
    + `${dHier < dMirror ? 'drops to' : 'sits at'} ${dHier.toFixed(3)}. That is a definition, not a `
    + `training failure.`);

  set('v-proto', `Filtered ranking against all ${n0(M.entities)} entities, which is the only `
    + `comparison the published numbers support.`);
  set('v-proto-warn', `It moves the headline by `
    + `${((W[best].eval.mrr / Math.max(W[best].eval.mrr_raw, 1e-9) - 1) * 100).toFixed(0)}% on its `
    + `own, so a number quoted without saying which protocol produced it says very little.`);

  set('closing-1',
    `The lesson of this section is unusual for this atlas, because it is a lesson about proofs `
    + `rather than about measurements. Almost everywhere else, the way to find out what a model can `
    + `do is to run it. Here, one line of algebra on the scoring function tells you in advance that `
    + `a whole class of facts cannot be represented, and the run goes and finds the parameters the `
    + `line was about: a relation vector ${tSym.toFixed(2)} long where the others average `
    + `${tAnti.toFixed(2)}, a rotation ${rSym.toFixed(3)} away from being its own inverse where the `
    + `others sit at ${rAnti.toFixed(2)}.`);

  set('closing-1b',
    `${compHolds && invHolds
      ? 'And the identities of the written graph came out the same way, which is the tidy version '
        + 'of this story.'
      : `But the other half of the page is the correction to that, and it is worth more than the `
        + `confirmation. On the written graph, where composition and inversion were put in by hand `
        + `and could have been read off the vectors just as easily, neither identity appeared. `
        + `Both kinds of statement are proofs about the same scoring function; only one of them is `
        + `a proof about where the optimiser ends up. Symmetry is a `
        + `<span class="bold">constraint</span>, so gradient descent has nowhere else to go and `
        + `arrives without being asked. Composition is a <span class="bold">capability</span>, so `
        + `it is available and free to ignore, and a run of this size ignores it. The useful `
        + `question to ask of an expressiveness result is therefore not "is it true" but "does it `
        + `forbid something or merely allow something", because only the first kind survives `
        + `contact with a finite budget.`}`);

  set('closing-2',
    `That closes the sixth branch of this atlas. The five articles of it share one move, which is `
    + `worth naming now that they are all in place: none of them describes an object, and all of `
    + `them describe a position. A book is where its readers put it, a node is where its `
    + `neighbours put it, an entity is where its relations put it. What the four earlier modules `
    + `spent their time on, extracting features from a thing so a model can look at it, does not `
    + `happen anywhere in this one.`);

  set('closing-3',
    `Where the atlas goes next is the other half of that sentence. Everything in this module `
    + `predicts, and prediction is not the only thing anybody wants from data: sooner or later `
    + `somebody asks whether the recommendation caused the purchase, whether a model's confidence `
    + `means anything, and what a model looked at when it decided. `
    + `<a href="../confounding/">The arrow you cannot see</a> starts that branch.`);

  set('ref-note',
    `WN18RR comes from the distribution that ships with the RotatE code, checked by digest `
    + `(<code>${M.digest}</code>): ${n0(M.train)} training facts, ${n0(M.valid)} for validation and `
    + `${n0(M.test)} for test, over ${n0(M.entities)} entities and ${M.relations} relations. All `
    + `three models are written in numpy with their gradients by hand, ${M.dim} dimensions `
    + `(${M.dim / 2} complex planes for RotatE), trained for ${M.epochs} passes with ${M.neg} `
    + `self-adversarial negatives per side and margin ${M.gamma}; these are small models at a small `
    + `budget, so the numbers are for comparing the three against each other rather than against a `
    + `leaderboard. Ranking is filtered against the union of train, validation and test, over `
    + `${n0(M.eval_triples)} test facts scored on both sides. The written graph has `
    + `${n0(P.meta.entities)} entities and ${n0(P.meta.triples)} triples over `
    + `${P.meta.relations} relations, one per pattern, from `
    + `<code>src/utils/generate_kg_data.py</code>.`);

  return IDS;
}
