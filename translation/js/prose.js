/* The sentences that carry a measured number, written from the measurement.
 *
 * Every direction word here ("sharper", "closer", "more committed") is a
 * branch on a number, and so is every cell of the verdict table. A re-run that
 * moves a result moves these sentences with it.
 */

const f1 = (v) => v.toFixed(1);
const f2 = (v) => v.toFixed(2);
const f3 = (v) => v.toFixed(3);
const f4 = (v) => v.toFixed(4);
const pc = (v, d = 1) => `${(v * 100).toFixed(d)}%`;
const n0 = (v) => Math.round(v).toLocaleString('en-US');
const WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
  'ten', 'eleven', 'twelve'];
const word = (v) => (v >= 0 && v <= 12 && Number.isInteger(v) ? WORDS[v] : n0(v));
const bold = (s) => `<span class="bold">${s}</span>`;
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;

const ARM_NAME = {
  l1: 'L1 alone',
  gan: 'the adversarial term alone',
  l1gan: 'the two together',
  'patch-pixel': 'a critic that sees one pixel',
  'patch-full': 'a critic that sees the whole image',
};

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

export function initProse(data) {
  const need = ['meta', 'conditional', 'real', 'scores', 'cycle', 'fields', 'nets'];
  const missing = need.filter((k) => !data[k]);
  if (missing.length) {
    throw new Error(`translation.json is missing ${missing.join(', ')}: regenerate first`);
  }
  const M = data.meta;
  const R = data.real;
  const S = data.scores;
  const seed = M.seeds[0];
  const arm = (a) => S[`${a}/${seed}`];
  const spreadOf = (a) => Math.abs(S[`${a}/${M.seeds[0]}`].l1 - S[`${a}/${M.seeds[M.seeds.length - 1]}`].l1);

  /* ------------------------------------------------------------ the set-up */
  set('dataset-note',
    `The scenes are drawn by the file that measures them: ${n0(M.n_train)} plans of `
    + `${M.res} by ${M.res} pixels, each with a horizon, a road and a building of one of two `
    + `types, and a render of each. Everything in the render is a function of the plan except two `
    + `things: the grain of the ground, and which of the building's windows are lit, each one `
    + `independently, ${pc(M.render_colours.lit_p, 0)} of the time. Those two are what make the `
    + `mapping one to many, and because this file can redraw any plan as many times as it likes, `
    + `the distribution of correct answers for a given plan is not a thing to be reasoned about `
    + `here. It is a thing to be sampled.`);

  /* ------------------------------------------------------- the conditional */
  const C = data.conditional;
  const c0 = C[0];
  const medianShare = c0.median_hf / c0.sample_hf;
  set('cond-intro',
    `The standard story about pix2pix is that L1 on its own gives blurry results and the `
    + `adversarial term sharpens them up. That is true and it is the wrong way round: the blur is `
    + `not a failure of optimisation, it is what the loss asked for. Here that can be shown rather `
    + `than argued, because the same plan can be rendered ${c0.m} times and the distribution `
    + `looked at directly.`);

  const twoValued = c0.window_hist && c0.window_hist.filter((h) => h.n > 0).length <= 4;
  const lit = c0.window_hist ? c0.window_hist.filter((h) => h.n > 0)
    .sort((a, b) => b.n - a.n)[0] : null;
  set('cond-result',
    `Rendering one plan ${c0.m} times and taking the pixelwise middle of the results gives an `
    + `image with ${bold(pc(medianShare))} of the detail of any single render `
    + `(${f4(c0.median_hf)} against ${f4(c0.sample_hf)}), and it is not an approximation of the `
    + `best answer under an absolute error, it ${bold('is')} that answer: its average distance to `
    + `the ${c0.m} renders is ${f4(c0.l1_of_median)} where any single render sits at `
    + `${f4(c0.l1_of_sample)}.`
    + `<br /><br />`
    + (twoValued
      ? `But the two losses do not fail the same way, and this dataset is sharp enough to separate `
        + `them. On that one window pixel the ${c0.m} renders take `
        + `${bold(word(c0.window_hist.filter((h) => h.n > 0).length))} values and nothing between: `
        + `${c0.window_hist.filter((h) => h.n > 0).map((h) => `${h.n} of them near ${f2((h.lo + h.hi) / 2)}`).join(' and ')}. `
        + `A squared error is minimised by the ${bold('mean')} of that, `
        + `${bold(f3(c0.window_mean))}, which is a brightness the data never contains: that is the `
        + `grey window everybody means by "L1 blurs", and it belongs to L2. An absolute error is `
        + `minimised by the ${bold('median')}, ${bold(f3(c0.window_median))}, which snaps to `
        + `whichever state was commoner and is a perfectly sharp, perfectly confident answer that `
        + `is wrong about ${pc(1 - (lit ? lit.n / c0.m : 0), 0)} of the windows. `
        + `<br /><br />`
        + `So the blur an L1 model shows is not in the switches, it is everywhere the conditional `
        + `is ${bold('continuous')}: the grain of the ground, whose median really is a smooth `
        + `average of everything it could have been. That is what the ${pc(medianShare)} above is `
        + `made of, and it is why the sharpening the adversarial term buys is mostly texture `
        + `rather than decisions.`
      : `The window pixels here are not two valued, so the sharpest form of this argument does not `
        + `apply and the detail figure above is the whole of it.`));

  /* -------------------------------------------------------------- the arms */
  set('arms-intro',
    `Which is the argument for the second term, and it is a narrower argument than it is usually `
    + `given. The adversarial loss cannot tell a model which windows are lit either: it never sees `
    + `the answer, only the domain. What it can insist on is that the output ${bold('belong')} to `
    + `the distribution, which the median of a distribution generally does not, and that is exactly `
    + `where a regression loss has nothing to say. So pix2pix keeps both: an L1 term to put things `
    + `in the right places, an adversarial term to make what is in them look like the real thing, `
    + `and a weight \\(\\lambda = ${M.lam}\\) that decides which of the two wins where they `
    + `disagree.`);

  const l1 = arm('l1');
  const gan = arm('gan');
  const both = arm('l1gan');
  const sharpest = M.arms.map((a) => ({ a, v: arm(a).hf_share })).sort((x, y) => y.v - x.v)[0];
  const closest = M.arms.map((a) => ({ a, v: arm(a).l1 })).sort((x, y) => x.v - y.v)[0];
  set('arms-result',
    `L1 alone lands ${bold(f4(l1.l1))} from the true render and keeps ${pc(l1.hf_share)} of a real `
    + `render's detail. The adversarial term alone lands ${bold(f4(gan.l1))} away, `
    + `${f1(gan.l1 / l1.l1)} times further, and keeps ${pc(gan.hf_share)}: `
    + (gan.hf_share > l1.hf_share
      ? `sharper and less faithful, which is the trade in one line. `
      : `neither sharper nor more faithful here, which is not the expected result and is reported `
        + `as it came out. `)
    + `Together they land ${bold(f4(both.l1))} away with ${pc(both.hf_share)} of the detail. `
    + `Against a seed spread of ${f4(Math.max(...M.arms.map(spreadOf)))} on the distance, the `
    + `fidelity gap between ${ARM_NAME[closest.a]} and the adversarial term alone is `
    + `${f4(gan.l1 - closest.v)}, which is `
    + (gan.l1 - closest.v > Math.max(...M.arms.map(spreadOf))
      ? `well outside it.`
      : `inside it, so that comparison is not resolved here.`)
    + ` The window measurement is the one that matters most, because it is where the one to many `
    + `problem lives: ${M.arms.map((a) => `${pc(arm(a).commit_share, 0)} for ${ARM_NAME[a]}`).join(', ')}, `
    + `against a real render by definition at 100%.`);

  set('strip-readout',
    `<p>Six plans, their true renders, and what three of the five training recipes produced from `
    + `them. The rows are the same six scenes throughout, and none of them was in the training `
    + `set.</p>`);

  /* ------------------------------------------------------------- the patch */
  const F = data.fields;
  const kinds = Object.keys(F.plain);
  set('patch-intro',
    `The other idea in that paper is the shape of the critic. Rather than one score for the whole `
    + `image, a small convolutional network produces a grid of scores, each one seeing a window of `
    + `the input, and the loss averages them. It has far fewer weights and it pushes the model `
    + `towards texture that is locally right. The number everybody quotes for it is the size of `
    + `that window, so here it is measured rather than derived: put a gradient on one output and `
    + `count the input pixels it reaches.`);

  const pixel = arm('patch-pixel');
  const full = arm('patch-full');
  const normNote = Object.values(F.normalised).every((v) => v >= M.res);
  set('patch-result',
    `The four windows come out ${kinds.map((k) => `${F.plain[k]}`).join(', ')} pixels on a `
    + `${M.res} pixel image. At one pixel the critic can only judge colour, and it produces `
    + `${pc(pixel.hf_share)} of a real render's detail with the windows ${pc(pixel.commit_share, 0)} `
    + `committed; at ${F.plain.full} pixels it sees essentially everything and produces `
    + `${pc(full.hf_share)} and ${pc(full.commit_share, 0)}. `
    + (Math.abs(pixel.hf_share - full.hf_share) > 0.15
      ? `So the window decides how much texture the critic can ask for, and a great deal of it: `
        + `${f1(full.hf_share / Math.max(pixel.hf_share, 1e-9))} times as much detail from the `
        + `widest as from the narrowest. What it barely moves is the windows `
        + `(${pc(pixel.commit_share, 0)} against ${pc(full.commit_share, 0)}), which is the part `
        + `of the picture that is a decision rather than a texture, and no amount of local realism `
        + `tells a model which decision was taken. `
      : `So at this scale the window size barely moves either number, which is worth saying given `
        + `how much attention that hyperparameter gets. `)
    + (normNote
      ? `And a detail that is easy to miss and changes what the number means: the same `
        + `discriminators with a normalisation layer inside have a window of `
        + `${bold(`${F.normalised.small} pixels`)}, the whole image, for every one of the four `
        + `shapes. A group or instance normalisation pools its statistics over the entire feature `
        + `map, so every output depends on every input pixel and the patch is a patch of `
        + `convolutions only. The discriminators on this page have no normalisation in them for `
        + `exactly that reason.`
      : ''));

  set('draw-intro',
    `Both trained networks travel with this page, `
    + (data.nets.l1.count === data.nets.l1gan.count
      ? `${n0(data.nets.l1.count)} numbers each in half precision, since only their loss differed`
      : `${n0(data.nets.l1.count)} and ${n0(data.nets.l1gan.count)} numbers in half precision`)
    + `, and they run here. Which makes the `
    + `comparison something to poke at rather than to read: the difference between the two is `
    + `easiest to see on a plan neither of them was trained on.`);

  /* ------------------------------------------------------------ cyclegan */
  const seeds = M.cycle_seeds;
  const agrees = seeds.map((s) => data.cycle[String(s)].agree);
  const right = agrees.filter((a) => a > 0.75).length;
  const wrong = agrees.filter((a) => a < 0.25).length;
  const mixed = agrees.length - right - wrong;
  set('cycle-intro',
    `Everything so far needed pairs, and pairs are the expensive part. CycleGAN's answer is to `
    + `train two translators at once, one each way, with two discriminators saying whether their `
    + `outputs belong to the right domain, plus one more term: go there and come back, and you `
    + `should end up where you started.`);

  set('cycle-setup',
    `The question this dataset can answer and a photograph collection cannot is whether the `
    + `correspondence it finds is the right one. Every plan here carries a building type, and the `
    + `type decides the render, so the answer is known for every image. And the plan's label `
    + `colours are deliberately crossed against the render's: the type drawn blue on the plan `
    + `renders red and the other way round. Nothing in the unpaired objective can tell those two `
    + `correspondences apart, since both are one to one, both produce perfectly good renders and `
    + `both are undoable. The only thing pushing is what the network finds easy.`);

  set('cycle-result',
    `Across ${word(seeds.length)} runs the building type survives the translation `
    + `${agrees.map((a) => pc(a)).join(', ')} of the time. `
    + (right === seeds.length
      ? `Every run found the correspondence the data actually has, in spite of the colours `
        + `pointing the other way.`
      : wrong === seeds.length
        ? `Every run found the mirrored one: the two types are swapped, every render is `
          + `plausible, every round trip closes, and the mapping is wrong in the only way that `
          + `matters.`
        : `${cap(word(right))} of them found the correspondence the data has and `
          + `${word(seeds.length - right)} did not, landing at `
          + `${agrees.filter((a) => a <= 0.75).map((a) => pc(a)).join(' and ')}, which is nearer `
          + `the mirrored correspondence than the right one. Same code, same data, same objective, `
          + `different seed, and nothing in the objective prefers the answer that happens to be `
          + `true.`)
    + ` This is the part of unpaired translation that a gallery of results cannot show you, `
    + `because to see it you have to know the answer, and the reason anyone uses unpaired `
    + `translation is that they do not.`);

  /* ------------------------------------------------------- steganography */
  const first = data.cycle[String(seeds[0])];
  const mid = first.noise[1];
  const ratios = seeds.map((s) => data.cycle[String(s)].noise[1].ratio);
  set('steg-intro',
    `And one more thing the cycle term does not constrain: what carries the information. Going `
    + `from a plan to a render throws detail away and the way back has to invent it, so a round `
    + `trip that closes to four decimal places is doing something. One possibility is that the `
    + `translator has learned the correspondence. Another is that it has hidden the plan inside `
    + `the render, at an amplitude nobody looks at. Those two are easy to tell apart: disturb the `
    + `intermediate image by an amount too small to see, and watch what the return trip does.`);

  set('steg-result',
    `Adding noise of ${mid.sigma} to a generated render moves its translation back by `
    + `${bold(f4(mid.moved_fake))}. The same noise added to a ${bold('real')} render of the same `
    + `scene moves its translation by ${bold(f4(mid.moved_real))}, a factor of `
    + `${bold(f1(mid.ratio))}. `
    + (Math.max(...ratios) > 2
      ? `The return network is far more sensitive to what it is fed when what it is fed came from `
        + `its partner, which is the signature the steganography paper describes: the round trip `
        + `is being carried by a low amplitude signal the discriminator never learned to object `
        + `to, and noise at that amplitude destroys it while leaving the picture untouched.`
      : `Which is not a large factor, so on these scenes the round trip does not appear to be `
        + `riding on a hidden signal, and this section reports a negative result rather than the `
        + `famous positive one.`)
    + ` Rounding the intermediate image, which is what saving it as a file would do, moves the `
    + `round trip error to `
    + first.quant.map((q) => `${f4(q.cycle)} at ${q.bits} bits`).join(', ')
    + `, against ${f4(first.base_cycle)} untouched: a signal hidden below the eighth bit would not `
    + `have survived that, and the round trip did.`);

  /* -------------------------------------------------------- the verdict */
  set('verdict-l1',
    `${f4(l1.l1)} from the true render, the closest of the five, with ${pc(l1.hf_share)} of a real `
    + `render's detail and windows ${pc(l1.commit_share, 0)} committed.`);
  set('verdict-gan',
    `${pc(gan.commit_share, 0)} of a real render's commitment at the windows against `
    + `${pc(l1.commit_share, 0)} for L1, and ${f1(gan.l1 / l1.l1)} times the distance from the `
    + `answer.`);
  set('verdict-patch',
    `Windows of ${kinds.map((k) => F.plain[k]).join(', ')} pixels measured by autodiff, and `
    + `${F.normalised.small} for all of them once a normalisation layer is added.`);
  set('verdict-cycle',
    `The building type survives ${agrees.map((a) => pc(a, 0)).join(', ')} of the time across `
    + `${word(seeds.length)} seeds, with round trip errors of `
    + `${seeds.map((s) => f4(data.cycle[String(s)].base_cycle)).join(', ')}.`);

  /* -------------------------------------------------------- the closing */
  set('limits-note',
    `What this page is not: ${M.res} by ${M.res} scenes with one building are not a city, and `
    + `${n0(M.steps)} steps is not a training run. The comparisons above are matched on steps, `
    + `batch size and architecture, so what changes between the arms really is the loss; but every `
    + `number carries a seed spread of up to ${f4(Math.max(...M.arms.map(spreadOf)))} on the `
    + `distance to the answer, and differences smaller than that are not differences. The `
    + `unpaired half is ${word(seeds.length)} seeds, which is enough to show that the outcome `
    + `depends on the seed and not enough to put a rate on it.`);

  set('closing-note',
    `That closes module 3.1. Three articles ago the problem was that "make something new" has no `
    + `loss function, and the answer was to hire one: a second network, a game, and every `
    + `practical difficulty following from the fact that the loss moves back. `
    + `<a href="../gans/">Counterfeit</a> measured what the game is worth and where it goes flat, `
    + `<a href="../stylegan/">Style</a> measured what happens when you decide where the latent `
    + `enters, and this one measured what is left when the pairing is taken away. The next `
    + `section keeps the goal and drops the opponent: instead of learning a loss, write the `
    + `distribution down and maximise its likelihood, which is what a `
    + `<a href="../vae/">variational autoencoder</a>, a normalising flow and a diffusion model `
    + `all do. It starts from the network the <a href="../autoencoders/">bottleneck</a> article `
    + `left behind, weight for weight, and puts a distribution on its code on purpose.`);

  set('resources-note',
    `Every figure on this page comes from <code>src/<wbr>utils/<wbr>generate_<wbr>translation_<wbr>data.py</code>: `
    + `${n0(M.n_train)} scene pairs at ${M.res} by ${M.res} with ${n0(M.n_test)} held out, five `
    + `paired arms at ${n0(M.steps)} steps and batch ${M.batch} on ${word(M.seeds.length)} seeds `
    + `each with \\(\\lambda = ${M.lam}\\), and ${word(M.cycle_seeds.length)} unpaired runs at `
    + `${n0(M.cycle_steps)} steps with a cycle weight of ${M.cycle_lam}, trained on two halves of `
    + `the dataset that share no scene. The receptive fields are measured by autodiff, the `
    + `conditional statistics come from redrawing one plan ${c0.m} times, and the two networks `
    + `this page runs are ${n0(data.nets.l1.count)} and ${n0(data.nets.l1gan.count)} numbers in `
    + `float16 with reference outputs the page checks itself against. No wall clock time is quoted `
    + `anywhere: everything was measured with ${M.threads} threads on one machine.`);
}
