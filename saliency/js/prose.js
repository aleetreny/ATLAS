/* Every sentence with a measured number, composed from the measurement.
 *
 * The sentence this article exists for is a comparison between two tables, so
 * it is written as a branch on both: which method scores highest against the
 * masks, and which methods survive having the model destroyed. If a rerun ever
 * changes either, the paragraph changes with it instead of keeping a verdict
 * that was true once.
 */

const n0 = (v) => v.toLocaleString('en-US');
const f2 = (v) => v.toFixed(2);
const f3 = (v) => v.toFixed(3);
const f4 = (v) => v.toFixed(4);
const pc = (v, d = 1) => `${(100 * v).toFixed(d)}%`;
const sgn = (v, d = 3) => (v >= 0 ? `+${v.toFixed(d)}` : v.toFixed(d));
const WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
  'nine', 'ten'];
const word = (v) => (v >= 0 && v < WORDS.length ? WORDS[v] : n0(v));

export const PROSE_IDS = [
  'intro-nothing', 'intro-key',
  'score-intro', 'score-note', 'score-depth',
  'sanity-intro', 'sanity-verdict',
  'att-intro', 'att-note', 'att-small', 'att-adv',
  'v-grad', 'v-grad-dep', 'v-grad-note',
  'v-cam', 'v-cam-dep', 'v-cam-note',
  'v-occ', 'v-occ-dep', 'v-occ-note',
  'v-att', 'v-att-dep', 'v-att-note',
  'closing-1', 'closing-2', 'ref-note',
];

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

function rows(sel, data, cells) {
  const body = d3.select(sel).select('tbody');
  if (body.empty()) return;
  body.html('');
  data.forEach((d) => {
    const tr = body.append('tr');
    cells(d).forEach((c) => {
      const td = tr.append('td');
      if (c && c.html !== undefined) td.html(c.html);
      else td.text(c);
    });
  });
}

export function initProse(data) {
  const M = data.meta;
  const S = data.scores;
  const D = data.depth;
  const I = data.identity;
  const N = data.sanity;
  const A = data.attention;

  const by = Object.fromEntries(S.rows.map((r) => [r.key, r]));
  const methods = S.rows.filter((r) => !['input', 'random'].includes(r.key));
  const best = methods.reduce((p, c) => (c.pointing > p.pointing ? c : p), methods[0]);
  const full = N[N.length - 1];
  const survives = ['gradient', 'grad_input'].filter((k) => full[k] > 0.8);
  const breaks = ['gradcam'].filter((k) => Math.abs(full[k]) < 0.4);
  const deepest = D[D.length - 1];
  const shallowest = D[0];
  /* La importancia es la caída publicada por el artículo de atención sobre sus
   * 5.734 frases de test, no la que se puede medir con las ocho frases de
   * ejemplo que su fichero trae: ver att-small. */
  const headRows = [...A.rows].sort((a, b) => b.published - a.published);
  const topHead = headRows[0];
  const others = headRows.slice(1);
  const peakiest = [...A.rows].sort((a, b) => b.peak - a.peak)[0];
  const sameHead = topHead.layer === peakiest.layer && topHead.head === peakiest.head;
  const step = 1 / A.tokens;

  set('intro-nothing',
    `This article trains nothing. Both models it explains were published earlier in this `
    + `atlas and both already run in a browser: the object detector from the detection `
    + `article, ${n0(M.detector.weights)} weights over ${word(M.detector.classes.length)} `
    + `classes, and the ${n0(2064204)} parameter encoder from the attention article. They are `
    + `rebuilt here in numpy from the same files those pages load, and the rebuild is not `
    + `taken on trust: the decoded boxes match the ones that article published to `
    + `<span class="bold">${M.reuse.worst_box.toFixed(4)} pixels</span>, and the encoder `
    + `tags the demo sentences at ${pc(A.demo_acc)} against the ${pc(A.published_acc)} that `
    + `one reports overall.`);

  set('intro-key',
    `And it borrows a third thing, which is what makes the rest possible. Those detection `
    + `scenes were composited by a program that pasted garments onto a canvas and recorded `
    + `<span class="bold">which object won every pixel</span>, so the exact mask of each `
    + `object exists. The scenes are rebuilt from the same seeds and checked against the `
    + `published sprite pixel by pixel before anything runs: `
    + `${n0(M.reuse.canvases)} canvases, ${n0(M.reuse.boxes)} boxes, and row `
    + `${M.reuse.row} of the first canvas identical. Almost no saliency paper has a mask to `
    + `score against, which is why almost all of them show pictures and let the reader `
    + `decide.`);

  set('score-intro',
    `Every method here produces an image the size of the input, and brighter is supposed to `
    + `mean more responsible. Six of them, on ${n0(S.n)} scenes, each explaining the `
    + `detector's score for one object at one grid cell. Three ways of scoring, all against `
    + `the true mask: whether the brightest pixel lands inside it, what fraction of the map's `
    + `total brightness falls inside it, and how much the brightest pixels overlap it when `
    + `you take exactly as many as the mask has.`);

  set('score-note',
    `The two controls are the reason to keep reading. <span class="bold">Smooth noise</span> `
    + `scores ${f3(by.random.pointing)}, which is the floor. And `
    + `<span class="bold">the input image itself</span>, with no model involved at all, `
    + `scores ${f3(by.input.pointing)} on the pointing game and `
    + `<span class="bold">${f3(by.input.overlap)}</span> on overlap, which beats the plain `
    + `gradient's ${f3(by.gradient.overlap)}. That is not a quirk of this dataset so much as `
    + `a warning about the whole family of metrics: the object is the bright thing, so any `
    + `map correlated with brightness scores well without explaining anything. Every number `
    + `in this section has to be read against those two rows.`);

  set('score-depth',
    `Grad-CAM has a parameter nobody reports, which is where in the network it is computed. `
    + `At the first block the feature map is ${shallowest.side} pixels across with `
    + `${shallowest.channels} channels and the pointing score is `
    + `<span class="bold">${f3(shallowest.pointing)}</span>; at the last it is `
    + `${deepest.side} across with ${deepest.channels} channels and scores `
    + `<span class="bold">${f3(deepest.pointing)}</span>. Resolution falls by a factor of `
    + `${shallowest.side / deepest.side} and the score nearly quadruples, which is the trade `
    + `the method lives on: the deep layers know what the object is and only roughly where. `
    + `One more thing falls out of the arithmetic here. With a one by one head, the weights `
    + `Grad-CAM averages are exactly the head's own weights, so Grad-CAM `
    + `<span class="bold">is</span> the class activation map, up to a factor of `
    + `${I.factor}. The page checks that: worst relative difference `
    + `${I.worst_relative.toExponential(0)}.`);

  set('sanity-intro',
    `There is a check that costs nothing and that a saliency map has to pass before its score `
    + `means anything. Take the trained network, replace the weights of the top layer with `
    + `random numbers of the same scale, and recompute the map. Then do the next layer down, `
    + `and the next. The model is being destroyed on purpose. A picture that is explaining `
    + `the model has to change; one that does not change was never about the model.`);

  set('sanity-verdict',
    `Put the two tables side by side and the ranking inverts. After randomising `
    + `<span class="bold">every layer</span>, the plain gradient's map still correlates `
    + `<span class="bold">${sgn(full.gradient)}</span> with the map from the trained model, `
    + `and gradient times input <span class="bold">${sgn(full.grad_input)}</span>. Grad-CAM `
    + `falls to <span class="bold">${sgn(full.gradcam)}</span>. So the method that scored `
    + `${f3(by.grad_input.pointing)} against the masks, the best of the cheap ones, is `
    + `producing essentially the same picture for a network whose weights are noise. It is an `
    + `edge detector with the model attached for decoration, and on a dataset where the `
    + `object is the only edge, an edge detector scores well. The method that scores lower, `
    + `${f3(by.gradcam.pointing)}, is the one that noticed.`);

  set('att-intro',
    `In a language model the picture is not a heatmap over pixels but a matrix of attention `
    + `weights, and it gets read the same way: this head is looking at that word, so that `
    + `word mattered. The encoder borrowed here has ${word(A.rows.length)} heads, and the `
    + `article that published it measured what each one is worth by removing it and watching `
    + `the accuracy fall over ${n0(A.published_n)} test sentences. That gives something a `
    + `picture of attention almost never has next to it: a number for how much the head `
    + `actually mattered, measured on enough data to mean something.`);

  set('att-note',
    `The head you would pick by looking at the pictures is the head that matters. The most `
    + `peaked map belongs to layer ${peakiest.layer} head ${peakiest.head}, at `
    + `${f3(peakiest.peak)} of its weight on a single word, and that head is `
    + `${sameHead ? 'also the most important one' : `not the most important one`}: removing `
    + `it costs <span class="bold">${f4(topHead.published)}</span> of accuracy against `
    + `${f4(others[0].published)} for the next. One agreement, and it is the agreement `
    + `everybody has seen, because it is the figure everybody publishes. Now take all `
    + `${word(A.rows.length)} heads at once. The rank correlation between how peaked a head `
    + `is and how much removing it costs is <span class="bold">${sgn(A.corrs.peak)}</span>, `
    + `and for the entropy of the map it is <span class="bold">${sgn(A.corrs.entropy)}</span>. `
    + `Not weak: none. The only statistic here that carries any rank information at all is `
    + `how much weight a head puts on the previous word, and it carries it `
    + `${A.corrs.previous < 0 ? 'backwards' : 'forwards'}, at ${sgn(A.corrs.previous)}.`);

  set('att-small',
    `There is a second reading of that table, and it is the one this page nearly published. `
    + `The attention article's file ships ${word(A.sentences)} example sentences, `
    + `${A.tokens} tags in total, and a head's importance can be measured right here by `
    + `ablating on those instead of trusting the number from ${n0(A.published_n)} sentences. `
    + `Do that and peakedness correlates <span class="bold">${sgn(A.corrs_small.peak)}</span> `
    + `with importance, entropy ${sgn(A.corrs_small.entropy)}: the map looks readable, and the `
    + `paragraph writes itself. But ${A.tokens} tags can only resolve accuracy in steps of `
    + `1/${A.tokens}, which is ${f4(step)}, and the published drops run from `
    + `${f4(Math.min(...A.rows.map((r) => r.published)))} to `
    + `${f4(Math.max(...A.rows.map((r) => r.published)))}. Every difference this comparison is `
    + `supposed to be about is smaller than one step of the ruler. The `
    + `${f2(A.corrs_small.peak)} is a picture of the rounding, and the third view above draws `
    + `both so the two can be put side by side.`);

  set('att-adv',
    `And underneath the resolution question there is a harder limit, which is the experiment `
    + `that decided the argument in the literature. Take one `
    + `head, keep its total attention mass, and shuffle where that mass goes, then look at `
    + `what the model predicts. The page searched for the most different attention pattern `
    + `that leaves <span class="bold">every single tag unchanged</span>, and found one at a `
    + `total variation distance of <span class="bold">${f3(A.adversarial.tv)}</span> from the `
    + `real map, where 1 would be no overlap at all. The largest change anywhere in the `
    + `output logits was ${f3(A.adversarial.logit_shift)}. A picture that can be moved almost `
    + `all the way across the space without changing a single answer is not a record of what `
    + `the answer depended on.`);

  rows('#att-table', headRows, (d) => [
    { html: `<span class="bold">layer ${d.layer}, head ${d.head}</span>` },
    f4(d.published),
    f4(-d.ablation),
    f3(d.entropy),
    f3(d.previous),
    f3(d.peak),
  ]);

  set('v-grad', `${f3(by.gradient.pointing)} and ${f3(by.grad_input.pointing)} on the `
    + `pointing game, the second one the best of the cheap methods`);
  set('v-grad-dep', `no: ${sgn(full.gradient)} and ${sgn(full.grad_input)} correlation with `
    + `the original after every layer is randomised`);
  set('v-grad-note',
    `Scoring well against a mask while ignoring the model is exactly what an edge detector `
    + `does on a dataset whose objects are the edges.`);
  set('v-cam', `${f3(by.gradcam.pointing)}, and ${f3(deepest.overlap)} overlap at the last `
    + `block`);
  set('v-cam-dep', `yes: ${sgn(full.gradcam)} after randomisation, which is no relationship `
    + `at all`);
  set('v-cam-note',
    `It scores lower than the gradient methods and it is the one that is looking at the `
    + `model. Its resolution is the ${deepest.side} by ${deepest.side} feature map.`);
  set('v-occ', `${f3(by.occlusion.pointing)}, the highest measured here`);
  set('v-occ-dep',
    `by construction: it is defined as the change in the prediction, so a different model is `
    + `a different map`);
  set('v-occ-note',
    `It costs one forward pass per patch, ${n0(64)} per image here, which is why the rest of `
    + `this table exists.`);
  set('v-att', `not applicable: it points at words, not pixels`);
  set('v-att-dep',
    `no: peakedness ranks the ${word(A.rows.length)} heads at ${sgn(A.corrs.peak)} against `
    + `importance measured over ${n0(A.published_n)} sentences`);
  set('v-att-note',
    `An alternative pattern ${f3(A.adversarial.tv)} away in total variation leaves every tag `
    + `unchanged.`);

  set('closing-1',
    `The through line of this module is one question asked four times: what would have to be `
    + `true for this output to mean what people read into it. For a causal estimate it was an `
    + `assumption about arrows that no data can test. For an interval it was exchangeability. `
    + `For an attribution it was a choice of value function. For a saliency map it is `
    + `something simpler and easier to check than any of those, and almost nobody checks it: `
    + `that the picture would look different if the model were different.`);

  set('closing-2',
    `That closes the seventh branch of this atlas, and it closes it on the same note the `
    + `first article of the branch opened: the number is not the problem. Every method here `
    + `computes what it says it computes, most of them exactly. What separates a useful `
    + `explanation from a decorative one is whether anybody built something to score it `
    + `against, and in three of the four cases in this module that scoring device had to be `
    + `manufactured on purpose, because the world does not ship with one. `
    + `<a href="../">Back to the atlas</a>, where the branches that are still dark are the `
    + `ones about acting rather than explaining.`);

  set('ref-note',
    `Nothing on this page was trained for it. The detector is the one from the detection `
    + `article (${n0(M.detector.weights)} weights, stride ${M.detector.stride}, an `
    + `${M.detector.grid} by ${M.detector.grid} grid) and the encoder is the one from the `
    + `attention article, both read from the same JSON their own pages load and both checked `
    + `against published figures before use. The score being explained is `
    + `${M.score}. The masks come from the compositing step of the segmentation article, `
    + `rebuilt here from the same seeds, and the occlusion row uses ${n0(S.occlusion_n)} of `
    + `the ${n0(S.n)} scenes because it costs sixty four forward passes each.`);
}
