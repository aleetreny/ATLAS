/* Every number in the running text, composed from the file at load. */
const n0 = (v) => Number(v).toLocaleString('en-US');
const f4 = (v) => Number(v).toFixed(4);
const mb = (v) => `${(v / (1024 * 1024)).toFixed(2)} MB`;

export const PROSE_IDS = [
  'opening-note', 'spec-intro', 'spec-note', 'rank-intro', 'rank-note',
  'alpha-intro', 'quant-intro', 'quant-note', 'mem-intro', 'verdict-intro',
  'verdict-lora', 'verdict-lora-warn', 'verdict-bitfit', 'verdict-bitfit-warn',
  'verdict-quant', 'verdict-quant-warn', 'verdict-double', 'verdict-double-warn',
  'closing-1', 'closing-2', 'ref-note',
];

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

export function initProse(data) {
  const M = data.meta;
  const S = data.spectra;
  const L0 = S.layers[0];
  const arms = data.arms;
  const loras = arms.filter((a) => a.rank != null).sort((a, b) => a.rank - b.rank);
  const full = arms.find((a) => a.arm === 'everything');
  const head = arms.find((a) => a.arm === 'head only');
  const bit = arms.find((a) => a.arm === 'biases plus head');
  const r8 = loras.find((a) => a.rank === 8) || loras[Math.floor(loras.length / 2)];
  const Q = data.quant;
  const mem = data.memory;
  const memFull = mem.rows.find((r) => r.arm === 'full fine tune');
  const memL = mem.rows.find((r) => r.arm === 'lora');
  const memQ = mem.rows.find((r) => r.arm === 'qlora');
  const qlo = data.qlora;
  const byRank = {};
  qlo.forEach((r) => { byRank[`${r.arm}|${r.rank}`] = r; });
  const rk = Math.max(...qlo.map((r) => r.rank));
  const fl = byRank[`float base|${rk}`];
  const nf = byRank[`nf4 base|${rk}`];
  const qCost = fl.mean - nf.mean;
  const qTie = Math.abs(qCost) <= fl.sd + nf.sd;
  const lowRank = S.layers.every((L) => L.delta.r90 < L.random.r90);
  const anyLow = S.layers.some((L) => L.delta.r90 < L.random.r90);
  const blocks = Q.blocks;
  const b64 = blocks.find((b) => b.block === 64) || blocks[Math.floor(blocks.length / 2)];
  const meanNf4 = Q.rows.reduce((s, r) => s + r.nf4, 0) / Q.rows.length;
  const meanInt4 = Q.rows.reduce((s, r) => s + r.int4, 0) / Q.rows.length;
  const nf4Wins = meanNf4 < meanInt4;
  const sweep = data.data_sweep;

  set('opening-note',
    `The model is the one the <a href="../distillation/">transfer and distillation article</a> `
    + `used, so the `
    + `two pages share their setup rather than describing similar ones: a perceptron of `
    + `${M.hidden.join(' and ')} hidden units and ${n0(M.params)} parameters, pretrained on `
    + `${n0(M.source_n)} digits and adapted to ${n0(M.target_n)} rows of clothing, scored on `
    + `${n0(M.test_n)} held out rows. It is a small model with big matrices, which is the `
    + `right shape for this question and the wrong shape for a few others; where that `
    + `matters it is said.`);

  set('spec-intro',
    `Fine tune the whole thing and subtract: the difference between the adapted matrix and `
    + `the pretrained one is a matrix of the same size, and its singular values say how many `
    + `directions the change actually used. This is the claim LoRA rests on and it is one `
    + `decomposition away.`);

  set('spec-note',
    `The curve on its own would mean nothing, which is why there are three. A random matrix `
    + `already concentrates some of its energy at the top, purely because singular values `
    + `come out sorted, so the honest question is not "how many directions hold ninety per `
    + `cent" but "how many fewer than a matrix that learned nothing". On the first layer the `
    + `update needs <span class="bold">${L0.delta.r90}</span> directions against `
    + `${L0.random.r90} for a random matrix of the same shape and the same norm and `
    + `${L0.base.r90} for the pretrained matrix itself. `
    + (lowRank
      ? `Every layer here comes out the same way, so the premise holds on this model.`
      : anyLow
        ? `Not every layer comes out that way, which is worth saying: the premise is a `
          + `measurement and not a law.`
        : `<span class="bold">No layer here comes out that way</span>, so on this model the `
          + `premise fails, and the sections that follow are measuring what an adapter does `
          + `when its justification does not hold.`)
    + ` And the update is small in the first place: its norm is `
    + `${(L0.rel * 100).toFixed(2)}% of the norm of the matrix it is added to. Adaptation is `
    + `a nudge. Fitting more rows of the new task makes it a slightly bigger nudge in `
    + `slightly more directions: from ${sweep[0].r90} directions at ${n0(sweep[0].n)} rows to `
    + `${sweep[sweep.length - 1].r90} at ${n0(sweep[sweep.length - 1].n)}.`);

  set('rank-intro',
    `Knowing the update is concentrated does not tell you that gradient descent through a `
    + `narrow slot will find the concentration. Those are two different claims and they are `
    + `usually reported as one, because the experiment that separates them needs the full `
    + `fine tune you were trying to avoid.`);

  set('rank-note',
    `Here it is available, so the ceiling can be drawn: truncate the real update to rank r `
    + `with a singular value decomposition and apply it. Nothing trained by gradient descent `
    + `can beat that at the same rank, because that is the best rank r approximation there `
    + `is. Reading the trained adapter against it says which of the two claims is doing the `
    + `work. The second view then puts every cheap method on the one axis they all compete `
    + `on: rank ${r8.rank} moves ${n0(r8.trainable)} parameters, `
    + `${(r8.trainable / full.trainable * 100).toFixed(2)}% of the network, and reaches `
    + `${f4(r8.mean)} against ${f4(full.mean)} for moving everything and ${f4(head.mean)} for `
    + `moving the head alone.`);

  set('alpha-intro',
    `One dial in the LoRA formula gets tuned as though it were capacity, and it is worth `
    + `three sentences because the algebra makes a falsifiable prediction about it. The `
    + `adapter contributes \\((\\alpha/r)BA\\), which is linear in \\(B\\), so multiplying `
    + `alpha multiplies the gradient that reaches \\(B\\) by exactly the same factor. That `
    + `is a learning rate wearing a hat. The test: sweep alpha with the learning rate fixed, `
    + `then sweep it again dividing the learning rate by the same factor. If the algebra is `
    + `the whole story, the second sweep is flat.`);

  set('quant-intro',
    `The other half of the acronym is about the frozen base. If nothing in it is going to `
    + `move, it does not need to be stored to full precision, and the question becomes how `
    + `to spend four bits per weight. Sixteen levels have to be placed somewhere, and the `
    + `three answers in use place them very differently.`);

  set('quant-note',
    `Uniform steps assume nothing. Floating point steps crowd near zero, on the assumption `
    + `that small weights need finer resolution. The quantile format places its levels at the `
    + `quantiles of a normal distribution, which is optimal exactly when the weights are `
    + `normal, so that assumption is checked in the table below on the actual weights rather `
    + `than asserted. Averaged over the layers the quantile format lands at `
    + `${f4(meanNf4)} relative error against ${f4(meanInt4)} for uniform, `
    + (nf4Wins ? `so it wins, ` : `so it does not win here, `)
    + `and the whole network squashed to four bits moves from ${f4(Q.base_acc)} to `
    + `${f4(Q.quant_acc)} before any adapter is trained on it. Then the block size, which is `
    + `the trade nobody states: each block of weights carries its own scale, so smaller `
    + `blocks fit better and cost more metadata. At ${b64.block} weights per block the real `
    + `price is ${b64.bits.toFixed(3)} bits per weight rather than four, and quantising the `
    + `scales themselves brings it to ${b64.bits_double.toFixed(3)}.`);

  set('mem-intro',
    `The reason any of this exists is a memory bill, and the bill is arithmetic rather than `
    + `benchmark. Adam keeps two moments per trainable parameter, and a gradient is one more `
    + `copy, so a trainable weight costs four times what a frozen one costs. That single `
    + `factor is why freezing the base saves nearly everything rather than saving a quarter.`);

  set('verdict-intro',
    `Four pieces, and they are independent: an adapter with no quantisation is a real recipe, `
    + `and a quantised base with full fine tuning on top is not.`);

  set('verdict-lora',
    `Rank ${r8.rank} reaches ${f4(r8.mean)} against ${f4(full.mean)} for full fine tuning, `
    + `moving ${(r8.trainable / full.trainable * 100).toFixed(2)}% of the parameters.`);
  set('verdict-lora-warn',
    `The adapter has to be added back before the model can be shipped, and the update it `
    + `finds is not the best one of its rank: the widest gap to the truncated ideal here is `
    + `${f4(Math.max(...loras.map((a) => {
      const c = data.ceiling.find((r) => r.rank === a.rank);
      return c ? c.acc - a.mean : 0;
    })))}.`);
  set('verdict-bitfit',
    `Moving the biases and the head reaches ${f4(bit.mean)} with ${n0(bit.trainable)} `
    + `parameters, against ${f4(head.mean)} for the head alone.`);
  set('verdict-bitfit-warn',
    `There are only ${n0(bit.trainable - head.trainable)} biases in this network, so this `
    + `arm has no dial: it is as small as it is and cannot be made bigger.`);
  set('verdict-quant',
    `An adapter of rank ${rk} on a four bit base reaches ${f4(nf.mean)} against `
    + `${f4(fl.mean)} on the full precision one.`);
  set('verdict-quant-warn',
    qTie
      ? `A difference of ${f4(Math.abs(qCost))}, inside the ${f4(fl.sd + nf.sd)} of seed `
        + `spread, for base weights that take ${(4 / 32 * 100).toFixed(0)}% of the storage.`
      : `Costs ${f4(qCost)}, larger than the ${f4(fl.sd + nf.sd)} of seed spread, in exchange `
        + `for base weights that take ${(4 / 32 * 100).toFixed(0)}% of the storage.`);
  set('verdict-double',
    `Brings the real cost of a four bit weight from ${b64.bits.toFixed(3)} bits to `
    + `${b64.bits_double.toFixed(3)} at ${b64.block} weights per block.`);
  set('verdict-double-warn',
    `It is metadata compression, so it does nothing to the error of the weights themselves `
    + `beyond the second rounding of the scales. The saving is `
    + `${((b64.bits - b64.bits_double) / b64.bits * 100).toFixed(1)}% of the storage, which `
    + `only matters once the first four bits are already paid.`);

  set('closing-1',
    `Put the pieces on one line and the arithmetic is the argument. Full fine tuning of this `
    + `network costs ${mb(memFull.total)} of weights, gradients and optimiser state; the `
    + `adapter of rank ${mem.rank} costs ${mb(memL.total)}, and with the frozen base at four `
    + `bits, ${mb(memQ.total)}. That is a factor of `
    + `${(memFull.total / memQ.total).toFixed(1)} on a model of ${n0(mem.params)} `
    + `parameters, and the factor grows with the model, because the part that shrinks is the `
    + `part that scales with the whole network while the part that stays is the adapter.`);

  set('closing-2',
    `Everything in this branch has now been chosen by a held out score: the dials in the `
    + `first article, the pipeline in the second, what to freeze in the third, the rank here. `
    + `<a href="../crossval/">The next article</a> asks what that score is, and it opens with `
    + `the only situation where the question has a definite answer: data generated by a `
    + `process that is written down, so the risk being estimated can be computed instead of `
    + `estimated.`);

  set('ref-note',
    `Everything on this page comes from `
    + `<span class="mono">src/<wbr>utils/<wbr>generate_<wbr>lora_<wbr>data.py</span> and the numpy kit in `
    + `<span class="mono">src/<wbr>utils/<wbr>tinynet.py</span>, seed ${M.seed}. The base is `
    + `${M.hidden.join(' and ')} hidden units trained on ${n0(M.source_n)} MNIST rows; every `
    + `adaptation uses the same ${n0(M.target_n)} Fashion-MNIST rows and ${M.epochs} epochs `
    + `of Adam, and every point is ${M.seeds} seeds. Ranks ${M.ranks.join(', ')}, alphas `
    + `${M.alphas.join(', ')} at rank ${M.alpha_rank}. The quantiser is blockwise absolute `
    + `maximum scaling exactly as bitsandbytes does it, with the sixteen levels of the `
    + `quantile format built from the normal quantiles rather than copied from a table, and `
    + `the bits per weight counted including the scales. The memory table is arithmetic on `
    + `the parameter counts of these networks: four bytes each for weights and gradients, `
    + `eight for the two Adam moments. No wall clocks anywhere on this page.`);
}
