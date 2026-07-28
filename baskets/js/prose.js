/* The sentences that carry a measured number, written from the measurement.
 *
 * Every comparison is a branch, and both of the tables that compare protocols
 * (the widget mines 24 items, the tables use 169) say which is which where the
 * numbers appear, because two counts from two protocols sitting next to each
 * other is an invitation to divide them.
 */

const n0 = (v) => v.toLocaleString('en-US');
const f3 = (v) => v.toFixed(3);
const pc = (v, d = 2) => `${(100 * v).toFixed(d)}%`;
const WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
  'ten', 'eleven', 'twelve'];
const word = (v) => (v >= 0 && v < WORDS.length ? WORDS[v] : n0(v));
const cap = (s) => s[0].toUpperCase() + s.slice(1);
const rule = (r) => `{${r.a.join(', ')}} &rarr; {${r.c.join(', ')}}`;

export const PROSE_IDS = [
  'intro-scale', 'intro-guard', 'scrolly-intro', 'step-0', 'step-1', 'step-2', 'step-3', 'step-4',
  'lattice-intro', 'lattice-note', 'sweep-note', 'tree-intro', 'tree-note',
  'rules-intro', 'rules-lift', 'trap-note', 'symmetry-note',
  'perm-intro', 'perm-note', 'wine-intro', 'wine-note', 'cost-note',
  'v-ap', 'v-ap-watch', 'v-fp', 'v-fp-watch', 'v-conf', 'v-conf-watch', 'v-lift', 'v-lift-watch',
  'closing-1', 'closing-2', 'ref-note',
];

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

/* 748288838313422294120286634350736906063837462003711 is not a number anybody
 * reads. Group it the way a person would say it. */
/* A 51 digit number grouped in threes is a 67 character token with no space in
   it, and a comma is not a line break opportunity, so on a 375px screen it does
   not wrap: it runs to 653px and drags the whole document sideways. A <wbr>
   after each comma lets it break where a reader would already pause, and
   nowhere else. */
function bigDigits(s) {
  return s.replace(/\B(?=(\d{3})+(?!\d))/g, ',').replace(/,/g, ',<wbr>');
}

export function initProse(data, facts) {
  const G = data.groceries;
  const A = data.apriori;
  const T = A.top;
  const R = data.rules;
  const P = data.permutation;
  const W = data.wine;

  /* ------------------------------------------------------------- the opening */
  set('intro-scale',
    `Counting is the easy part. One pass over ${n0(G.n)} real grocery baskets says that whole `
    + `milk is in ${pc(G.all_counts[0] / G.n)} of them and that whole milk and other vegetables `
    + `are in the same basket <span class="bold">${n0(data.rules.pair_count ?? 0)}</span> times, `
    + `and a pass is a pass. The problem is how many questions there are to ask. These baskets `
    + `hold <span class="bold">${G.items}</span> distinct item categories, so the number of sets `
    + `you could ask about is 2<sup>${G.items}</sup> &minus; 1, which is `
    + `<span class="bold">${G.total_subsets_digits} digits long</span>: `
    + `${bigDigits(G.total_subsets)}. There is no machine, and there is not going to be one.`);

  set('intro-guard',
    `Both run in this page. The baskets travel as one bitset per item, so the support of any set `
    + `of items is one AND and one population count, and that is fast enough for the page to `
    + `enumerate a whole lattice while you drag a slider. Apriori runs here with its pruning `
    + `counted, the prefix tree is built here three ways, and every rule below is derived here. `
    + `The guards are the strong kind: on ${word(14)} items the page counts all `
    + `${n0(2 ** 14 - 1)} subsets exhaustively and checks that Apriori found exactly the same `
    + `frequent sets, and that no subset of a frequent set is rarer than it, before it draws `
    + `anything.`);

  /* ------------------------------------------------------------- the scrolly */
  set('scrolly-intro',
    `Five items, and every set you could make from them: `
    + `${facts.names.slice(0, -1).join(', ')} and ${facts.names[facts.names.length - 1]}, the `
    + `five commonest things in the file. Thirty-one sets, which is a number a person can hold. `
    + `The rest of the article is about what happens when there are ${G.items} items instead of `
    + `five.`);
  set('step-0',
    `Five items, laid out by how many of them each set holds: five sets of one, ten of two, ten `
    + `of three, five of four, and one of five. ${cap(word(facts.subsets))} in all, and the shape `
    + `is the thing to notice: the lattice is fattest in the middle, which is exactly where the `
    + `counting would go if nobody stopped it.`);
  set('step-1',
    `The support of a set is how many baskets contain all of it, and one pass over the database `
    + `gives every one of these numbers. Singly they run from ${n0(Math.min(...facts.singles))} `
    + `to ${n0(Math.max(...facts.singles))} baskets out of ${n0(G.n)}; by the time you are `
    + `looking at pairs the largest is already down in the hundreds.`);
  set('step-2',
    `Join each set to the sets one item larger that contain it, and the subsets stop being a list `
    + `and become a structure. Every path upwards adds one item and can only make the support `
    + `smaller, which sounds like a triviality and is the only reason any of this is computable.`);
  set('step-3',
    `Call a set frequent when its support reaches some minimum, here `
    + `${n0(facts.minCount)} baskets, which is ${pc(facts.minCount / G.n, 0)}. Filled is `
    + `frequent, hollow is not: <span class="bold">${word(facts.frequent)}</span> of the `
    + `${facts.subsets} make it. Everything else is noise as far as this threshold is concerned, `
    + `and the threshold is a choice nobody can make for you.`);
  set('step-4',
    `A set can only be frequent if every one of its subsets is, so the moment a set fails, `
    + `everything above it in the lattice fails too and need never be counted. On these five items `
    + `that is <span class="bold">${word(facts.cut)}</span> of the ${facts.subsets} sets skipped, `
    + `which is ${pc(facts.cut / facts.subsets, 0)}. On ${G.items} items it is the difference `
    + `between a ${G.total_subsets_digits} digit number and an afternoon.`);

  /* ------------------------------------------------------------- the lattice */
  set('lattice-intro',
    `A basket that contains B contains A as well, so A cannot be rarer. Turned around, and this `
    + `is the form the algorithm uses: <span class="bold">if A is not frequent then nothing `
    + `containing A can be</span>, so an entire cone of the lattice can be discarded on the `
    + `strength of one count. The widget below does not take that on trust. It enumerates every `
    + `subset of the items you pick, counts each one exactly, and then compares what Apriori found `
    + `against what exhaustion found, every time you move a control.`);

  set('lattice-note',
    `The same experiment on all ${G.items} items is in the table below, and it says the same `
    + `thing at a scale no browser can enumerate. At `
    + `${pc(A.min_support, 0)}, which is ${n0(A.min_count)} baskets, Apriori generated `
    + `${n0(A.work.generated)} candidates, threw <span class="bold">${n0(A.work.pruned)}</span> of `
    + `them away on the subset test without ever counting them, and counted `
    + `${n0(A.work.counted)} in ${word(A.work.passes)} passes over the database. It found `
    + `${n0(A.work.frequent)} frequent sets, the largest holding ${word(A.longest)} items. The `
    + `part of the lattice it could conceivably have had to look at, every set of up to `
    + `${word(A.longest)} items out of ${G.items}, holds ${bigDigits(A.reachable)} sets: it `
    + `counted ${(100 * A.work.counted / Number(A.reachable)).toFixed(4)}% of that.`);

  const sweep = A.sweep;
  const st = document.querySelector('#sweep-table tbody');
  if (st) {
    st.innerHTML = sweep.map((r) => `<tr><td>${pc(r.support, 2)}</td><td>${n0(r.count)}</td>`
      + `<td>${n0(r.frequent)}</td><td>${n0(r.counted)}</td><td>${n0(r.pruned)}</td></tr>`).join('');
  }
  const top = sweep[0];
  const bot = sweep[sweep.length - 1];
  set('sweep-note',
    `Two things in that table are worth more than the numbers. The first is that the threshold is `
    + `not a tuning knob, it is the question: dropping it from ${pc(top.support, 0)} to `
    + `${pc(bot.support, 1)} takes the answer from ${n0(top.frequent)} frequent sets to `
    + `${n0(bot.frequent)}, a factor of ${(bot.frequent / top.frequent).toFixed(0)}, and nothing `
    + `about the data changed. The second is that <span class="bold">the pruning grows faster than `
    + `the work</span>: at ${pc(top.support, 0)} the algorithm throws away `
    + `${(top.pruned / top.counted).toFixed(3)} candidates for every one it counts, and at `
    + `${pc(bot.support, 1)} it throws away ${(bot.pruned / bot.counted).toFixed(3)}. The lower `
    + `the threshold, the harder downward closure works, which is the opposite of how a shortcut `
    + `usually behaves. (Every row of that table is capped at sets of `
    + `${word(bot.sizes.length >= 4 ? 4 : bot.sizes.length)} items so that the bottom of the sweep `
    + `terminates, which is why its ${pc(A.min_support, 0)} row and the paragraph above it do not `
    + `have to agree.)`);

  /* ---------------------------------------------------------------- the tree */
  set('tree-intro',
    `Apriori attacks the number of candidates. FP-Growth attacks the database. Sort each basket's `
    + `items into one fixed order, insert it into a prefix tree, and baskets that begin the same `
    + `way share the same nodes: the ${n0(data.tree.best.slots)} item slots these ${n0(G.n)} `
    + `baskets hold of the ${n0(data.tree.items)} items frequent at ${pc(A.min_support, 0)} `
    + `become ${n0(data.tree.best.nodes)} nodes, and after that the transactions are never read `
    + `again.`);

  const tt = document.querySelector('#tree-table tbody');
  if (tt) {
    tt.innerHTML = data.tree.rows.map((r) => `<tr><td>${r.order}`
      + `${r.nodes === data.tree.best.nodes ? ' <span class="bold">(the algorithm\'s own)</span>' : ''}`
      + `</td><td>${n0(r.nodes)}</td><td>${n0(r.slots)}</td><td>${f3(r.ratio)}</td></tr>`).join('');
  }
  const best = data.tree.best;
  const worst = data.tree.worst;
  set('tree-note',
    `The ordering is worth measuring rather than assuming, and it turns out the textbook is `
    + `right: putting the commonest items nearest the root gives `
    + `<span class="bold">${n0(best.nodes)}</span> nodes and it is the smallest of the `
    + `${word(data.tree.rows.length)} orders tried, while the worst, `
    + `${worst.order}, needs ${n0(worst.nodes)}, which is `
    + `<span class="bold">${(worst.nodes / best.nodes).toFixed(3)} times</span> as many for `
    + `exactly the same data. The reason is one sentence: a prefix is only shared if the items at `
    + `the front are ones many baskets have, and putting the rare items first guarantees that `
    + `almost nothing is shared. Even a random order does better than the reverse, at `
    + `${n0(data.tree.rows.find((r) => r.order === 'a random order').nodes)} nodes, which says `
    + `the effect is not a small correction.`);

  /* --------------------------------------------------------------- the rules */
  set('rules-intro',
    `A frequent set is not a rule. {bread, butter} being common says nothing about direction, and `
    + `a rule points somewhere. At ${pc(A.min_support, 0)} support and `
    + `${pc(A.min_conf, 0)} confidence these ${n0(G.n)} baskets yield `
    + `<span class="bold">${n0(R.n)}</span> rules over all ${G.items} items.`);

  set('rules-lift',
    `Confidence is how often B follows A. Lift is how that compares with how often B happens `
    + `anyway, and it is the one that can be below one. The highest confidence rule in the file `
    + `is ${rule(R.best_conf)} at ${pc(R.best_conf.confidence, 1)}, and its lift is `
    + `${f3(R.best_conf.lift)}; the highest lift rule is ${rule(R.best_lift)} at `
    + `${f3(R.best_lift.lift)}, with a confidence of only ${pc(R.best_lift.confidence, 1)}. `
    + `Those are two different rules, and a list sorted one way does not resemble a list sorted `
    + `the other.`);

  const trap = R.trap;
  set('trap-note',
    trap
      ? `The trap has a name and a shape. <span class="bold">${n0(R.below_one)}</span> of the `
        + `${n0(R.n)} rules have lift below one, and the worst of them is `
        + `${rule(trap)}: confidence <span class="bold">${pc(trap.confidence, 2)}</span>, which `
        + `sounds like a finding, and lift ${f3(trap.lift)}, which is not. `
        + `${cap(trap.c[0])} is in ${pc(R.trap_prior)} of all baskets to begin with, so a rule `
        + `that reaches ${pc(trap.confidence, 2)} has made its own conclusion `
        + `<span class="bold">less</span> likely than knowing nothing at all. Confidence cannot `
        + `see that, because confidence never looks at how common the consequent is. Every rule `
        + `whose consequent is a staple has this shape, and staples are exactly what a basket `
        + `dataset is full of.`
      : `At these thresholds no rule has lift below one; lower the confidence in the widget above `
        + `and they appear.`);

  const asym = R.worst_asym;
  set('symmetry-note',
    asym
      ? `Lift fixes that and breaks something else: <span class="bold">it is symmetric</span>. `
        + `Of the ${n0(R.n)} rules, ${n0(R.pairs)} have their own reverse in the list, and across `
        + `every one of those pairs the two lifts agree to ${R.lift_gap.toExponential(1)}, which `
        + `is machine precision and not a measurement. The confidences do not: the widest gap is `
        + `${rule(asym)} at ${pc(asym.conf_ab, 1)} against the reverse at `
        + `${pc(asym.conf_ba, 1)}, a difference of ${pc(asym.gap, 1)} on a pair whose lift is `
        + `${f3(asym.lift)} in both directions. So lift can tell you that two things go together `
        + `and cannot tell you which way to read it, and confidence can tell you which way to read `
        + `it and cannot tell you whether it means anything. Neither is the answer on its own, `
        + `which is why both are in every library and why every practical recipe quotes both.`
      : `Lift is symmetric, which no pair in this list happens to demonstrate.`);

  /* --------------------------------------------------------- the permutation */
  set('perm-intro',
    `There is one more question that almost nobody asks of a pile of rules: how many of them `
    + `would have turned up in data with no associations in it at all? It is answerable, and `
    + `cheaply. Shuffle each item's presence independently across the ${n0(G.n)} baskets. Every `
    + `item keeps exactly the support it had, so the marginals are untouched and every threshold `
    + `still means the same thing; what is destroyed is any relationship between items. Then mine `
    + `it again, at the same thresholds, ${word(P.shuffles)} times.`);
  const pt = document.querySelector('#perm-table tbody');
  if (pt) {
    pt.innerHTML = `<tr><td>the real baskets</td><td>${n0(P.real_frequent)}</td>`
      + `<td>${n0(P.real_rules)}</td></tr>`
      + `<tr><td>${word(P.shuffles)} shuffles, mean</td><td>${P.mean_frequent.toFixed(1)} `
      + `&plusmn; ${P.sd_frequent.toFixed(1)}</td><td>${P.mean_rules.toFixed(1)} &plusmn; `
      + `${P.sd_rules.toFixed(1)}</td></tr>`
      + `<tr><td>the worst shuffle</td><td>${n0(P.max_frequent)}</td><td>${n0(P.max_rules)}</td></tr>`;
  }
  set('perm-note',
    `The shuffled data is not empty, and that is the point. `
    + `<span class="bold">${P.mean_rules.toFixed(1)} rules on average</span> survive the same `
    + `thresholds in data where every association has been destroyed, against `
    + `${n0(P.real_rules)} in the real baskets: `
    + `${pc(P.mean_rules / P.real_rules, 0)} of the harvest is the search talking to itself. `
    + `The frequent sets are worse, ${P.mean_frequent.toFixed(1)} against `
    + `${n0(P.real_frequent)}, which is ${pc(P.mean_frequent / P.real_frequent, 0)}, because a `
    + `pair of common items clears a ${pc(A.min_support, 0)} threshold on frequency alone. `
    + `Nothing here is broken: the algorithm is doing exactly what it promised, which is to find `
    + `every set above a threshold. It is the reading of the output that needs the correction, `
    + `and this test costs ${word(P.shuffles)} re-runs and gives a floor to compare against. `
    + `(Every row of that table is capped at sets of ${word(P.capped_at)} items, real and shuffled `
    + `alike, so the comparison is like for like.)`);

  /* ---------------------------------------------------------------- the wine */
  set('wine-intro',
    `Groceries are sparse: the average basket holds ${G.avg_basket.toFixed(2)} of ${G.items} `
    + `items, so almost every pair of items never co-occurs at all. Most tables are the opposite. `
    + `Take the ${n0(W.n)} bottles of wine this site has been carrying since its first article, `
    + `replace every measurement by which third of its range it falls in, and each bottle becomes `
    + `a basket of exactly <span class="bold">${W.basket_size}</span> items out of `
    + `${W.items.length}: the same construction `
    + `<a href="../assumptions/">the clustering article</a> used for k-modes, reproduced here `
    + `${W.shared && W.shared.gap === 0 ? 'exactly, code for code' : 'to within rounding'}.`);
  const wt = document.querySelector('#wine-table tbody');
  if (wt) {
    const rows = [
      ['baskets', n0(G.n), n0(W.n)],
      ['distinct items', n0(G.items), n0(W.items.length)],
      ['items per basket', `${G.avg_basket.toFixed(2)} on average, ${G.max_basket} at most`,
        `exactly ${W.basket_size}`],
      ['density', pc(G.avg_basket / G.items), pc(W.basket_size / W.items.length)],
      ['minimum support used', pc(A.min_support, 0), pc(W.min_support, 0)],
      ['frequent sets found', n0(A.work.frequent), n0(W.frequent)],
      ['candidates counted', n0(A.work.counted), n0(W.counted)],
      ['candidates pruned unseen', n0(A.work.pruned), n0(W.pruned)],
    ];
    wt.innerHTML = rows.map((r) => `<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td></tr>`).join('');
  }
  const wineTop = W.top && W.top.length ? W.top[0] : null;
  set('wine-note',
    `Same algorithm, and nothing alike in behaviour. The wine is `
    + `${(W.basket_size / W.items.length / (G.avg_basket / G.items)).toFixed(1)} times denser, so `
    + `a threshold that leaves ${n0(A.work.frequent)} frequent sets in the groceries at `
    + `${pc(A.min_support, 0)} needs to be ${pc(W.min_support, 0)} here just to keep the answer `
    + `to ${n0(W.frequent)} sets, and the pruning has almost nothing to do: `
    + `${n0(W.pruned)} candidates thrown away against ${n0(W.counted)} counted, where the `
    + `groceries throw away ${(A.work.pruned / A.work.counted).toFixed(2)} for every one counted. `
    + `In a dense table almost every candidate really is frequent, so there is nothing to prune `
    + `and the ${G.total_subsets_digits} digit problem comes straight back. `
    + (wineTop
      ? `The strongest rule it finds is ${rule(wineTop)} at lift ${f3(wineTop.lift)}, which is a `
        + `fact about chemistry rather than about shopping, and reads as one.`
      : ''));

  /* --------------------------------------------------------------- the cost */
  const cost = data.cost.rows;
  const ct = document.querySelector('#cost-table tbody');
  if (ct) {
    ct.innerHTML = cost.map((r) => `<tr><td>${n0(r.items)}</td>`
      + `<td>${r.digits <= 18 ? bigDigits(r.subsets) : `a ${r.digits} digit number`}</td>`
      + `<td>${n0(r.pairs)}</td><td>${n0(r.upto3)}</td></tr>`).join('');
  }
  set('cost-note',
    `The number to keep in view is the one at the top of the page, and the table says why it is `
    + `never the number anybody pays. Sets of up to three items out of ${G.items} is `
    + `${n0(cost.find((r) => r.items === 169).upto3)}, which a laptop counts without complaint; `
    + `every subset of those same ${G.items} items is ${G.total_subsets_digits} digits. Both `
    + `algorithms on this page are ways of never leaving the first column, and the threshold is `
    + `what keeps them there: it is not a preference, it is the thing that makes the problem `
    + `finite.`);

  /* ------------------------------------------------------------- the verdict */
  set('v-ap',
    `Every step is a count you can look at, and the pruning is provably lossless: on the `
    + `${word(T.n_items)} commonest items, where all ${n0(T.subsets)} subsets were counted `
    + `exhaustively, it found the same ${n0(T.frequent)} frequent sets from `
    + `${n0(T.counted)} counts, which is ${pc(T.counted / T.subsets, 2)} of the lattice.`);
  set('v-ap-watch',
    `The threshold, which is the question rather than a setting: ${pc(A.min_support, 0)} gives `
    + `${n0(A.sweep.find((r) => r.support === 0.01).frequent)} sets and ${pc(0.001, 1)} gives `
    + `${n0(A.sweep[A.sweep.length - 1].frequent)}. And one database pass per level, which is `
    + `${word(A.work.passes)} here and more when the sets get longer.`);
  set('v-fp',
    `One compression of the database and no candidate generation at all: `
    + `${n0(best.slots)} item slots become ${n0(best.nodes)} nodes, ${f3(best.ratio)} slots per `
    + `node, and everything after that is done on the tree.`);
  set('v-fp-watch',
    `The ordering: the same data costs ${(worst.nodes / best.nodes).toFixed(3)} times as many `
    + `nodes with the items in the reverse order. And the tree has to fit in memory, which `
    + `${f3(best.ratio)} slots per node makes easy on sparse baskets and a dense table like the `
    + `wine would not.`);
  set('v-conf',
    `The number people actually want: how often the conclusion follows. The best rule here reaches `
    + `${pc(R.best_conf.confidence, 1)}.`);
  set('v-conf-watch',
    trap
      ? `Common consequents. ${n0(R.below_one)} of the ${n0(R.n)} rules have lift below one, and `
        + `the worst of them has ${pc(trap.confidence, 1)} confidence pointing at something that `
        + `happens ${pc(R.trap_prior, 1)} of the time anyway.`
      : `Common consequents: confidence never looks at how likely the conclusion was to begin `
        + `with.`);
  set('v-lift',
    `The one measure here that knows the difference between a rule and a coincidence: one is the `
    + `neutral value, and ${n0(R.below_one)} rules in this file sit below it.`);
  set('v-lift-watch',
    `It is symmetric, to ${R.lift_gap.toExponential(1)} across the ${n0(R.pairs)} reversible pairs `
    + `measured here, so it cannot tell you which way round a rule should be read. And it says `
    + `nothing about whether the rule would have appeared in shuffled data, which is what the `
    + `permutation test is for.`);

  /* ------------------------------------------------------------- the closing */
  set('closing-1',
    `What this branch has that none of the others in module 2 do is an <span class="bold">exact `
    + `answer</span>. k-means finds a local optimum, t-SNE finds a different map every time, and `
    + `an autoencoder finds whatever the gradient led it to. Apriori finds `
    + `<span class="bold">every</span> set above the threshold, provably, and this page checks `
    + `that claim against exhaustive enumeration rather than repeating it. The price is that the `
    + `question has to be small enough to have an exact answer, and making it small enough is what `
    + `the threshold is for. Lower it far enough and the ${G.total_subsets_digits} digit number `
    + `comes back.`);
  set('closing-2',
    `That closes module 2. Six articles on clustering, five on dimensionality reduction, two on `
    + `anomalies and this one on baskets, and the thread through all fourteen is the same: every `
    + `method is an assumption with a search bolted on, the assumption decides what it can see at `
    + `all, and the search is what you actually get. Here the assumption is that co-occurrence is `
    + `the only relationship there is, and the search is exact; and even with an exact search over `
    + `an honest count, ${pc(P.mean_rules / P.real_rules, 0)} of what comes out survives having `
    + `the data shuffled. What comes next is a different question altogether. Everything in this `
    + `module has been about finding structure that was already in the data; module 3 is about `
    + `learning the distribution well enough to <span class="bold">make more of it</span>. It `
    + `starts <a href="../vae/">where the bottleneck article left off</a>, by taking that `
    + `article's own decoder, sampling it, and asking what it takes to make the samples worth `
    + `looking at.`);

  set('ref-note',
    `The baskets are the Groceries dataset that ships with the R package arules: ${n0(G.n)} `
    + `transactions over ${G.items} item categories from one month of a grocery store's point of `
    + `sale, kept outside the repository and checked by digest (${G.digest.slice(0, 16)}) so that `
    + `a mirror changing underneath shows up as a failure rather than as a different article. The `
    + `wine is the scikit-learn copy of the UCI file, binned to thirds `
    + `${W.shared && W.shared.gap === 0 ? 'and reproducing the codes the clustering article ships exactly'
      : 'as the clustering article binned it'}. The permutation test uses numpy's PCG64 at the `
    + `seed recorded in the generator. scikit-learn ${data.meta.sklearn}, scipy ${data.meta.scipy}, `
    + `numpy ${data.meta.numpy}.`);
}
