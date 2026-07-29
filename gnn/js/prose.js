/* Every number on this page, composed from the file it was measured into. */
const IDS = ['intro-1', 'intro-2', 'intro-3', 'arms-intro', 'arms-note', 'live-intro',
  'homo-intro', 'homo-note', 'depth-intro', 'depth-note', 'att-intro', 'att-note',
  'ind-intro', 'ind-note', 'verdict-intro', 'v-gcn', 'v-gcn-warn', 'v-hetero',
  'v-hetero-warn', 'v-sage', 'v-sage-warn', 'v-gat', 'v-gat-warn',
  'closing-1', 'closing-2', 'closing-3', 'ref-note'];

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

const n0 = (v) => Number(v).toLocaleString('en-US');
const pc = (v, d = 1) => `${(100 * v).toFixed(d)}%`;

export function initProse(data) {
  const M = data.meta;
  const C = data.cora;
  const H = data.homophily;
  const D = data.depth;
  const A = data.attention;
  const I = data.inductive;

  const bestDepth = D.trained.reduce((a, x) => (x.acc > a.acc ? x : a));
  const deepest = D.trained[D.trained.length - 1];
  const graphBeatsFeatures = C.label_prop.acc > C.mlp.acc;
  const lowH = H[0];
  const highH = H[H.length - 1];
  const attGap = A.ratio - A.uniform_ratio;
  const archSpread = Math.max(C.gcn.acc, C.sage.acc, C.gat.acc)
    - Math.min(C.gcn.acc, C.sage.acc, C.gat.acc);
  const normSpread = data.norms.sym.acc - data.norms.none.acc;

  set('intro-1',
    `Everything this atlas has classified so far arrived as independent rows. A wine, a digit, a `
    + `sentence: whatever it was, the model saw it alone. This module is about the case where the `
    + `rows are connected, and the whole family of methods comes out of one decision about what to `
    + `do with the connection.`);

  set('intro-2',
    `The decision is: before applying any weights, replace each row with a mixture of itself and `
    + `its neighbours. `
    + `$$H^{(l+1)} = \\sigma\\!\\left(\\tilde{A}\\, H^{(l)} W^{(l)}\\right), \\qquad `
    + `\\tilde{A} = \\tilde{D}^{-1/2} (A + I) \\tilde{D}^{-1/2}$$ `
    + `That is a graph convolutional layer in full. \\(A\\) is who cites whom, the identity keeps `
    + `each node's own row in the mixture, and the two square roots of the degree stop a paper with `
    + `a hundred citations from shouting over one with two. Everything else on this page is a `
    + `consequence of that line, including both of the ways it goes wrong.`);

  set('intro-3',
    `The graph is Cora: <span class="bold">${n0(M.nodes)}</span> machine learning papers, `
    + `<span class="bold">${n0(M.edges)}</span> citations between them, a bag of `
    + `${n0(M.features)} words for each, and seven topics. Only `
    + `<span class="bold">${M.train}</span> nodes carry a label, twenty per topic, and `
    + `${n0(M.test)} are scored: that scarcity is the point, because the reason to look at the `
    + `citations at all is that there are not enough labels to go around. `
    + `<span class="bold">${pc(M.homophily)}</span> of the citations join two papers on the same `
    + `topic, and that number turns out to be the one that decides everything.`);

  set('arms-intro',
    `Before any architecture, two models that are not architectures. One reads the words and never `
    + `looks at a citation. The other never looks at a word: it puts a one on each labelled node `
    + `and repeatedly averages along the edges, which is a method from 2003 with no parameters at `
    + `all. Running both is the only way to know which of the two inputs a graph model is actually `
    + `getting its accuracy from.`);

  set('arms-note',
    `${graphBeatsFeatures
      ? `The order of those two controls is the finding: on this dataset the citations alone `
        + `(${pc(C.label_prop.acc)}) beat the words alone (${pc(C.mlp.acc)}). Cora is a graph `
        + `problem with features attached rather than a feature problem with a graph attached, and `
        + `a paper reporting only the second control would make its model look better than it is.`
      : `The words alone (${pc(C.mlp.acc)}) beat the citations alone (${pc(C.label_prop.acc)}), so `
        + `most of what a graph model has here comes from the features.`} `
    + `And the second view is worth a minute: swapping the symmetric normalisation for plain `
    + `averaging or for no normalisation at all moves the number by `
    + `${(100 * normSpread).toFixed(1)} points, against ${(100 * archSpread).toFixed(1)} points `
    + `between the three architectures. The part of the model that is not learned is doing more `
    + `work than the part that is.`);

  set('live-intro',
    `Here is the mixing step itself, on a real piece of Cora. Nothing is trained: each round `
    + `replaces every node's value with the average of itself and its neighbours, which is `
    + `\\(\\tilde{A}H\\) with the weights left out. Watch the labelled nodes reach the unlabelled `
    + `ones, and then watch what happens if you keep going.`);

  set('homo-intro',
    `Everything above assumes that a citation means agreement. On Cora it usually does: `
    + `${pc(M.homophily)} of edges join two papers of the same topic. But that is a property of the `
    + `dataset, not of graphs, and a graph where it is false makes the mixing step actively `
    + `harmful. On a real graph the question cannot be asked, because you get the homophily you are `
    + `given, so these graphs are written by the generator with the classes decided first and the `
    + `share of same class edges as a knob.`);

  set('homo-note',
    `The useful way to read that chart is as a condition rather than a warning. Averaging over `
    + `neighbours is worth doing when the neighbours carry the same label often enough to beat the `
    + `noise it adds, and here that threshold is measurable: at ${pc(lowH.measured)} homophily the `
    + `GCN scores ${pc(lowH.gcn)} against ${pc(lowH.mlp)} for ignoring the graph entirely, and at `
    + `${pc(highH.measured)} it is ${pc(highH.gcn)} against ${pc(highH.mlp)}. Social networks and `
    + `citation networks are usually well above the threshold. Fraud graphs, where the whole point `
    + `is that a fraudulent account transacts with ordinary ones, are usually below it.`);

  set('depth-intro',
    `The second failure is on the other axis. A two layer network sees two hops; three layers, `
    + `three hops. The obvious move is to stack more, and everyone who tries it finds the same `
    + `thing, which has a clean explanation and a number attached to it.`);

  set('depth-note',
    `Best at <span class="bold">${bestDepth.layers}</span> layers `
    + `(${pc(bestDepth.acc)}), falling to ${pc(deepest.acc)} at ${deepest.layers}. The explanation `
    + `is in the second view and it needs no model at all: repeated averaging on a connected graph `
    + `converges to a fixed point where every node holds the same value, so the spread between `
    + `neighbours decays towards zero whether or not weights are being fitted. After `
    + `${D.smoothing[D.smoothing.length - 1].steps} rounds it has fallen by a factor of `
    + `${(D.smoothing[0].energy / Math.max(D.smoothing[D.smoothing.length - 1].energy, 1e-12)).toExponential(1)}. `
    + `A node's representation stops being about the node, and a classifier reading it has nothing `
    + `left to separate.`);

  set('att-intro',
    `Attention is the natural response to the observation that not every neighbour deserves the `
    + `same weight. Instead of one over the degree, learn a weight per edge from the two ends. The `
    + `question this page can ask, and most cannot, is whether the learned weights are better in a `
    + `sense that has a definition: every node's true topic is known, so an edge joining two papers `
    + `of the same topic is one the model ought to prefer.`);

  set('att-note',
    `${Math.abs(attGap) < 0.01
      ? `The comparison the bars make is not flattering, and it reproduces what the literature `
        + `found once anybody checked: on a homophilous benchmark the attention ends up nearly `
        + `where the fixed normalisation already was. Its accuracy here `
        + `(${pc(C.gat.acc)} against ${pc(C.gcn.acc)}) is within the seed spread of the `
        + `GCN's, which is what a mechanism that has learned the degree looks like.`
      : `The learned weights do separate the two kinds of edge, by ${attGap.toFixed(3)} more than `
        + `the degree alone, and the accuracy follows: ${pc(C.gat.acc)} against ${pc(C.gcn.acc)}.`} `
    + `The place where attention has room is the other kind of graph: where the neighbours disagree, `
    + `a mechanism that can put a negative or a near zero weight on an edge is the only thing on `
    + `this page that could survive, which is exactly where the homophily dial says the fixed `
    + `operators fall over.`);

  set('ind-intro',
    `One thing left, and it is the one that decides whether any of this can be deployed. Everything `
    + `above is transductive: the test nodes were in the graph while the model trained, unlabelled `
    + `but present, their citations part of every average. A recommender or a fraud system does not `
    + `work that way. The node arrives after the model exists.`);

  set('ind-note',
    `So here is that experiment. <span class="bold">${n0(I.hidden_nodes)}</span> test nodes are `
    + `deleted from the graph entirely, edges included, the model is trained on what is left, and `
    + `then the nodes are put back and classified. GraphSAGE reaches `
    + `<span class="bold">${pc(I.inductive.sage.unseen)}</span> on the nodes it never saw against `
    + `${pc(I.inductive.sage.seen)} on the ones it did; the GCN, `
    + `${pc(I.inductive.gcn.unseen)} against ${pc(I.inductive.gcn.seen)}. `
    + `${I.inductive.sage.unseen > I.inductive.gcn.unseen
      ? `The separate weight for a node's own features is what makes the difference: it means the `
        + `model has learned a function of a neighbourhood rather than a table of node vectors.`
      : `On this graph the two are within noise of each other, so the inductive gap is about the `
        + `operator rather than about the separate self weight.`} `
    + `And sampling, which is what makes this affordable on a graph with a billion edges, costs `
    + `less than it looks: keeping ${I.sampling[0].k} neighbour per node scores `
    + `${pc(I.sampling[0].acc)} (&plusmn;${(100 * I.sampling[0].sd).toFixed(1)} across draws) `
    + `against ${pc(I.full)} for the whole neighbourhood, and by `
    + `${I.sampling[I.sampling.length - 1].k} it is `
    + `${pc(I.sampling[I.sampling.length - 1].acc)}.`);

  set('verdict-intro',
    `Four situations, and the second one is the one people skip.`);

  set('v-gcn', `${pc(C.gcn.acc)} against ${pc(C.mlp.acc)} for the same features with the edges `
    + `thrown away, on ${M.train} labels.`);
  set('v-gcn-warn', `Two layers. ${bestDepth.layers === 2 ? 'Three is already worse and' : ''} `
    + `${deepest.layers} layers is ${pc(deepest.acc)}, because a deep stack is a long average.`);

  set('v-hetero', `Nothing. At ${pc(lowH.measured)} homophily the GCN scores ${pc(lowH.gcn)} `
    + `against ${pc(lowH.mlp)} for ignoring the graph.`);
  set('v-hetero-warn', `Measure the homophily of your graph before choosing the family. It is one `
    + `line of code and it decides the question.`);

  set('v-sage', `${pc(I.inductive.sage.unseen)} on nodes that were not in the graph during `
    + `training, and it survives keeping ${I.sampling[0].k} neighbour each.`);
  set('v-sage-warn', `Sampling adds variance you have to report: `
    + `&plusmn;${(100 * I.sampling[0].sd).toFixed(1)} points across draws at the smallest sample.`);

  set('v-gat', `${pc(C.gat.acc)}, ${Math.abs(C.gat.acc - C.gcn.acc) <= C.gcn.sd
    ? 'inside the seed spread of the GCN' : 'outside the seed spread of the GCN'}.`);
  set('v-gat-warn', `Its weights prefer same topic edges by ${A.ratio.toFixed(3)} against `
    + `${A.uniform_ratio.toFixed(3)} for one over the degree, which learns nothing.`);

  set('closing-1',
    `The shape of this family is easier to hold than its literature suggests. There is one fixed `
    + `operation, mixing a row with its neighbours, and everything else is a variation on it: which `
    + `mixture (the normalisation), how many rounds (the depth), whether the weights are fixed or `
    + `learned (attention), and whether the node keeps a channel of its own (GraphSAGE). Two `
    + `numbers decide whether any of it will work, and both can be computed before training `
    + `anything: how many edges join two nodes of the same class, and how many rounds it takes for `
    + `the averaging to erase the differences.`);

  set('closing-2',
    `What every model on this page assumes is that an edge is an edge. The mixing matrix has one `
    + `number per pair, so "cites", "is cited by" and "shares an author with" would all go into the `
    + `same average. On the graphs people actually keep in databases that is thrown away `
    + `information, and often the most interesting part of it.`);

  set('closing-3',
    `The last article of this module gives the edge a type, and asks what a vector can hold about a `
    + `relation. It turns out to be a question with closed form answers: some scoring functions `
    + `provably cannot represent a symmetric relation, others provably cannot compose two, and the `
    + `predictions those proofs make are visible in the trained parameters. `
    + `<a href="../kg-embeddings/">A relation is a movement</a> is next.`);

  set('ref-note',
    `Cora comes from the Planetoid distribution, checked by digest (<code>${M.digest}</code>), with `
    + `the standard split: ${M.train} labelled nodes, ${M.val} for validation and ${n0(M.test)} for `
    + `test. Its features are binary word indicators, ${pc(M.feature_density, 2)} of them on; the `
    + `average degree is ${M.degree_mean.toFixed(2)} and the largest is ${M.degree_max}. Every `
    + `model here is written in numpy with its gradients by hand, ${M.hidden} hidden units, Adam, `
    + `weight decay 5e-4, early stopping on the validation split, and ${M.seeds} seeds per arm with `
    + `the spread reported. The homophily dial uses written graphs of 1,000 nodes and average `
    + `degree 10, with the class of every node decided before the first edge exists. The spread `
    + `between neighbours is the Dirichlet energy of the representation, normalised by its own `
    + `size. From <code>src/utils/generate_gnn_data.py</code>.`);

  return IDS;
}
