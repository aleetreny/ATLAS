/* Every figure on this page, composed from the file it was measured into. */
const n0 = (v) => v.toLocaleString('en-US');
const pc = (v, d = 1) => `${(v * 100).toFixed(d)}%`;

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

export function initProse(data) {
  const M = data.meta;
  const P = data.policies;
  const E = data.epsilon;
  const T = data.etc;
  const H = data.horizon;
  const G = data.gaps;
  const D = data.decay;
  const R = data.reference;
  const best = Object.keys(P).reduce((a, b) => (P[b].final < P[a].final ? b : a),
    Object.keys(P)[0]);
  const bestArm = M.arms.indexOf(Math.max(...M.arms));
  const eps = E.rows.reduce((a, b) => (b.regret < a.regret ? b : a), E.rows[0]);
  const flips = D.rows.find((d) => d.log_r2 > d.linear_r2);
  const worstGap = G.rows.reduce((a, b) => (b.etc > a.etc ? b : a), G.rows[0]);

  set('opening-note',
    `A bandit is the smallest problem with that shape. There are ${M.arms.length} machines, each `
    + `pays one or nothing with a probability of its own, and you get ${n0(M.horizon)} pulls. `
    + `There is no state, no sequence and nothing to plan: the only decision is which machine, and `
    + `the only difficulty is that finding out costs the same thing you are trying to collect. `
    + `Everything in the rest of this module is this problem with something added back.`);

  set('setup-intro',
    `The five probabilities are ${M.arms.join(', ')}. They are written down rather than estimated, `
    + `and that is what makes this measurable: the cost of a pull is known exactly, so what a `
    + `policy lost can be added up rather than approximated. That quantity has a name, regret, and `
    + `it is the difference between what the best machine would have paid and what you actually `
    + `collected.`);

  set('setup-note',
    `Machine ${bestArm + 1} is the best at ${M.arms[bestArm]}, and machine `
    + `${M.arms.indexOf(Math.min(...M.arms)) + 1} the worst at ${Math.min(...M.arms)}. The two `
    + `closest are ${M.gaps.filter((g) => g > 0).sort((a, b) => a - b)[0]} apart, which is the `
    + `number that decides how hard this instance is. A policy that pulled the best machine every `
    + `time would regret zero; one that pulled at random would regret `
    + `${(M.gaps.reduce((a, b) => a + b, 0) / M.arms.length * M.horizon).toFixed(1)}. Everything `
    + `below sits between those two.`);

  set('scrolly-step-0',
    `One pull each, which is all any policy can do before it knows anything. The bars are the `
    + `estimate after those ${M.arms.length} pulls; the dashed lines are the truth, which nobody `
    + `playing ever sees. At this point the estimates are all 0 or 1.`);

  set('scrolly-step-1',
    `Whichever machine won its first pull now has an estimate of 1, so a policy that always takes `
    + `the best estimate takes that one, wins or loses, and updates. The danger is not that it is `
    + `wrong; it is that being wrong is self sealing. A machine it stops pulling never gets `
    + `another sample, so its estimate is frozen at whatever bad luck it had.`);

  set('scrolly-step-2',
    `This run explores ${pc(R.config.eps, 0)} of the time, so eventually an accident sends it to `
    + `the machine it had written off. That is the whole mechanism, and it is worth being precise `
    + `about it: the accident is not a good decision, it is the absence of a decision.`);

  set('scrolly-step-3',
    `After ${R.config.pulls} pulls it has spent ${R.counts[bestArm]} of them on the best machine `
    + `and regrets ${R.regret}. Its estimate of the best machine is `
    + `${R.means[bestArm].toFixed(3)} against a truth of ${M.arms[bestArm]}, and its estimate of `
    + `the machine it pulled least is ${R.means[R.counts.indexOf(Math.min(...R.counts))].toFixed(3)} `
    + `on ${Math.min(...R.counts)} pulls, which is the other half of the trade: the machines you `
    + `do not pull are the ones you are least sure about.`);

  set('scrolly-note',
    `One run says nothing, so here are five policies over ${n0(M.runs)} runs each, all on the same `
    + `five machines with the same ${n0(M.horizon)} pulls.`);

  set('policy-note',
    `The order is not the interesting part; the reason for it is. Always taking the best estimate `
    + `regrets <span class="bold">${P.greedy.final}</span>, and the column that explains it is the `
    + `next one: it ends up believing the wrong machine is best in `
    + `<span class="bold">${P.greedy.wrong_arm} of ${n0(M.runs)}</span> runs. Exploring `
    + `${pc(0.1, 0)} of the time cuts that to ${P.eps_10.wrong_arm} and the regret to `
    + `${P.eps_10.final}; exploring ${pc(0.01, 0)} of the time is barely better than not exploring `
    + `at all (${P.eps_01.wrong_arm} wrong, ${P.eps_01.final}). And the best of the five here is `
    + `<span class="bold">${best === 'etc' ? 'explore then commit' : best}</span> at `
    + `${P[best].final}, which is the crudest policy on the list: try everything a fixed number of `
    + `times, then stop thinking. It wins because ${n0(M.horizon)} pulls is short, and it is the `
    + `first hint that every answer in this article is an answer about a horizon.`);

  set('dials-intro',
    `Each of those policies has one number in it, and all three of those numbers are usually `
    + `copied from somewhere. Swept, each one shows a different way of being wrong.`);

  set('dials-note',
    `Three things come out of those sweeps. First, a fixed exploring rate buys a straight line: `
    + `the tax is paid at every pull regardless of what has been learnt, predicted at `
    + `${E.rows[E.rows.length - 2].predicted_slope} per pull for a rate of `
    + `${E.rows[E.rows.length - 2].eps} and measured at `
    + `${E.rows[E.rows.length - 2].measured_slope} over the back half of the run. The best rate `
    + `measured here is ${eps.eps}, and it is the best rate <span class="bold">for this `
    + `horizon</span> only. Second, the formula for how long to explore first, the one that gets `
    + `quoted, picks a length ${T.sweeps.map((s) => s.ratio).join(', ')} times longer than what `
    + `works, and paying that costs `
    + `${T.sweeps.map((s) => (s.regret_at_predicted / s.best_regret).toFixed(1)).join(', ')} times `
    + `the regret at the three gaps swept. It is not wrong, it minimises an upper bound rather `
    + `than the regret, and a loose bound gives a setting that is loose by the same factor. `
    + `Third, a decaying rate does buy a logarithm rather than a line, but only past a constant: `
    + `below ${flips ? flips.c : 'the values swept'} the regret still fits a straight line better `
    + `(${D.rows[0].linear_r2} against ${D.rows[0].log_r2} at c = ${D.rows[0].c}), and above it `
    + `the logarithm wins (${D.rows[D.rows.length - 1].log_r2} against `
    + `${D.rows[D.rows.length - 1].linear_r2}). The constant the theory asks for is about `
    + `${n0(D.required_c)}, far above where the fit flips, because it is a sufficient condition `
    + `and not a necessary one. <span class="bold">And it is written in terms of the smallest gap, `
    + `which is exactly the thing nobody knows before they start.</span>`);

  set('gap-intro',
    `Which leaves one question that the five machines above cannot answer, because they are one `
    + `instance. Is a bandit hard when its machines are close together? The obvious answer is yes `
    + `and it is wrong.`);

  set('gap-note',
    `With two machines and the gap swept, the policy that stops exploring has an easy case at `
    + `<span class="bold">each end</span>: a gap of ${G.rows[0].delta} costs ${G.rows[0].etc}, `
    + `because if the machines are the same it does not matter which you pick, and a gap of `
    + `${G.rows[G.rows.length - 1].delta} costs ${G.rows[G.rows.length - 1].etc}, because a `
    + `difference that large shows up in a handful of pulls. The expensive case is in the middle, `
    + `at ${worstGap.delta}, big enough to matter and small enough to hide, where it ends on the `
    + `wrong machine ${worstGap.etc_wrong} times in ${worstGap.of}. Theory predicts that peak sits `
    + `around one over the square root of the horizon, ${G.predicted_delta} here: the same order, `
    + `not the same number. The fixed rate has no peak at all, because its tax does not depend on `
    + `what it has learnt, so its worst case is the gap the other policy finds easiest.`);

  set('verdict-intro',
    `Nothing on this page knows how uncertain it is. Every policy here either explores blindly or `
    + `stops exploring on a schedule decided in advance, and the table is what that costs.`);

  set('verdict-etc',
    `crude and hard to beat at a short horizon: ${P.etc.final} against ${P.eps_10.final} for a `
    + `fixed rate over ${n0(M.horizon)} pulls`);
  set('verdict-etc-warn',
    `the length is the whole policy, and the formula for it picks something `
    + `${T.sweeps[0].ratio} to ${T.sweeps[T.sweeps.length - 1].ratio} times too long`);
  set('verdict-eps',
    `it never locks on: ${P.eps_10.wrong_arm} of ${n0(M.runs)} runs end on the wrong machine `
    + `against ${P.greedy.wrong_arm} for pure greed`);
  set('verdict-eps-warn',
    `the tax is forever: ${E.rows[E.rows.length - 2].measured_slope} of regret per pull at rate `
    + `${E.rows[E.rows.length - 2].eps}, so the total is a straight line`);
  set('verdict-decay',
    `a rate that falls like one over the pull count turns that line into a logarithm, fitted at `
    + `${D.rows[D.rows.length - 1].log_r2} against ${D.rows[D.rows.length - 1].linear_r2}`);
  set('verdict-decay-warn',
    `only above a constant of about ${n0(D.required_c)} for this gap, and the gap is what you do `
    + `not know`);
  set('verdict-greedy',
    `it is the one policy here that can be wrong forever: ${P.greedy.wrong_arm} of ${n0(M.runs)}`);
  set('verdict-greedy-warn',
    `and it looks fine on average until you read the spread, ${P.greedy.std} on a mean of `
    + `${P.greedy.final}`);

  set('closing-note',
    `Every policy on this page is blind in the same way. None of them knows which of its `
    + `estimates is shaky and which is solid, so exploring means picking at random and stopping `
    + `means deciding in advance. That is why the constants keep depending on the gap: without a `
    + `sense of its own uncertainty, a policy cannot tell an unlucky machine from a bad one. `
    + `There is a floor under all of this, worked out by Lai and Robbins: on these five machines `
    + `no reasonable policy can regret less than ${M.bound} times its share of the logarithm, `
    + `which at ${n0(M.horizon)} pulls is a statement about the eventual slope rather than about `
    + `this horizon. <a href="../thompson-ucb/">The next article</a> gives the policy a measure `
    + `of its own uncertainty, in two different ways, and gets within reach of that floor.`);

  set('resources-note',
    `Everything was measured by <code>src/utils/generate_bandits_data.py</code> with seed `
    + `${M.seed}: ${n0(M.runs)} runs of ${n0(M.horizon)} pulls for the five policy comparison, `
    + `1,500 runs per point for the sweeps, and 1,000 for the horizon fits. The machines pay one `
    + `or nothing with probabilities ${M.arms.join(', ')}. Regret is the exact quantity, computed `
    + `from those probabilities rather than from the rewards drawn. The reference run in the page `
    + `is ${R.config.pulls} pulls at a rate of ${R.config.eps} from seed ${R.config.seed}.`);
}
