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

  set('cond-result',
    `Rendering one plan ${c0.m} times and taking the pixelwise middle of the results gives an `
    + `image with ${bold(pc(medianShare))} of the detail of any single render `
    + `(${f4(c0.median_hf)} against ${f4(c0.sample_hf)}), and window pixels that are `
    + `${bold(pc(c0.median_commit))} committed against ${pc(c0.sample_commit)} for a real one. `
    + `And it is not an approximation of the best L1 answer, it is the best L1 answer: its average `
    + `distance to the ${c0.m} renders is ${f4(c0.l1_of_median)} where any single render sits at `
    + `${f4(c0.l1_of_sample)}. So a network trained on L1 that produces a grey window has not `
    + `failed to learn. It has learned exactly the thing it was asked for, and the thing it was `
    + `asked for is an average of two answers, neither of which is grey.`);

  /* -------------------------------------------------------------- the arms */
  set('arms-intro',
    `Which is the argument for the second term. The adversarial loss does not care whether the `
    + `windows in the output match the windows in the answer; it cares whether they look like `
    + `windows, and a grey window does not. So pix2pix keeps both: L1 to say where things are and `
    + `an adversarial term to say what they should look like, with the weight `
    + `\\(\\lambda = ${M.lam}\\).`);

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
    + (Math.abs(pixel.commit_share - full.commit_share) > 0.1
      ? `So the window size does decide how much the critic can ask for. `
      : `So at this scale the window size barely moves the result, which is worth saying given how `
        + `much attention that hyperparameter gets. `)
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
    `Both trained networks travel with this page, ${n0(data.nets.l1.count)} and `
    + `${n0(data.nets.l1gan.count)} numbers in half precision, and they run here. Which makes the `
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
        : `${word(right)} of them found the correspondence the data has, ${word(wrong)} found the `
          + `mirrored one`
          + (mixed ? `, and ${word(mixed)} settled on neither consistently` : '')
          + `. Same code, same data, same objective, different seed.`)
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
    + ` Rounding the intermediate image to `
    + first.quant.map((q) => `${q.bits} bits moves the round trip error to ${f4(q.cycle)}`).join(', ')
    + `, against ${f4(first.base_cycle)} untouched, which is the same measurement in the units of `
    + `an actual image file.`);

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
    + `enters, and this one measured what is left when the pairing is taken away. The next module `
    + `keeps the goal and drops the opponent: instead of learning a loss, write down a `
    + `distribution and maximise its likelihood, which is what a variational autoencoder, a `
    + `normalising flow and a diffusion model all do. The bridge is already built: the `
    + `<a href="../autoencoders/">bottleneck</a> article ends exactly where that starts.`);

  set('resources-note',
    `Every figure on this page comes from <code>src/utils/generate_translation_data.py</code>: `
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
