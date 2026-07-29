/* Every number on this page, composed from the file it was measured into. */
const IDS = ['opening-note', 'dial-intro', 'trade-intro', 'trade-note', 'held-intro',
  'live-intro', 'verdict-intro', 'verdict-obey', 'verdict-obey-warn', 'verdict-real',
  'verdict-real-warn', 'verdict-held', 'verdict-held-warn', 'closing-1', 'closing-2', 'ref-note'];

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

const n0 = (v) => Number(v).toLocaleString('en-US');
const pc = (v, d = 1) => `${(100 * v).toFixed(d)}%`;

export function initProse(data) {
  const M = data.meta;
  const S = data.sweep;
  const H = data.held;
  const bestObey = S.reduce((a, r) => (r.obeys > a.obeys ? r : a));
  const bestReal = S.reduce((a, r) => (Math.abs(r.mmd2) < Math.abs(a.mmd2) ? r : a));
  const zero = S.find((r) => r.scale === 0);
  const last = S[S.length - 1];
  const split = bestObey.scale !== bestReal.scale;

  set('opening-note',
    `The pipeline underneath is the previous article's, unchanged and checked: the same `
    + `autoencoder, whose round trip scores ${data.ceiling.mmd2} here against the `
    + `${data.ceiling.published} that article published for the same bottleneck. So the ceiling `
    + `is the same ceiling and everything new on this page is the conditioning. One thing is `
    + `worth flagging before any of it: obedience is scored with the shared judge's `
    + `<span class="bold">label</span> against the label that was asked for, never with its `
    + `confidence. <a href="../ebm/">The energy article</a> measured that this judge's confidence `
    + `is largely a measure of ink, and the guard says so again here (${data.ink.corr}). Which `
    + `label it picks is a different and much steadier question.`);

  set('dial-intro',
    `Ten labels asked for, ${data.strips.per_label} draws of each, at four settings of the dial, `
    + `with real digits of the same labels underneath for scale. At zero the label is ignored `
    + `entirely and the model produces whatever it likes: it happens to answer correctly `
    + `${pc(zero.obeys)} of the time, which is roughly the ten percent you get from guessing.`);

  set('trade-intro',
    `Both of the things you wanted are measurable, so both are on the chart. Obedience is the `
    + `fraction of ${n0(M.asked)} requests answered with the right digit. Realism is the same two `
    + `sample statistic every article in this module has used, against the same floor. `
    + `${split
      ? `They do not peak in the same place, and that is the whole of it.`
      : `On this model they peak in the same place, so there is nothing to trade.`}`);

  set('trade-note',
    `${split
      ? `Obedience is best at <span class="bold">${bestObey.scale}</span> (${pc(bestObey.obeys)}) `
        + `and realism at <span class="bold">${bestReal.scale}</span> (${bestReal.mmd2}). Past `
        + `the second, every further turn buys obedience with realism: by ${last.scale} the `
        + `samples sit ${(Math.abs(last.mmd2) / Math.abs(bestReal.mmd2)).toFixed(1)} times `
        + `further from real digits than at their best, while obeying `
        + `${last.obeys < bestObey.obeys ? 'less' : 'more'} than at ${bestObey.scale}.`
      : `Both peak at ${bestObey.scale}, so this page reports that rather than manufacturing a `
        + `tension it did not measure.`} `
    + `${bestObey.obeys > data.real.obeys
      ? `And there is a number in that table that should stop you. At ${bestObey.scale} the model `
        + `gives the requested digit ${pc(bestObey.obeys)} of the time, while the same judge names `
        + `the right digit for a <i>real</i> held out one only ${pc(data.real.obeys)} of the time. `
        + `Obeying better than reality does not mean beating reality. It means drawing the `
        + `textbook version of each digit, and the variety column says so plainly: `
        + `${bestObey.variety} against ${data.real.variety} for real ones.`
      : `Obedience tops out at ${pc(bestObey.obeys)}, under the ${pc(data.real.obeys)} the judge `
        + `manages on real digits.`}`);

  set('held-intro',
    `Which invites a question with a clean experiment behind it: is the dial making the model `
    + `<i>know</i> more about the label, or just lean on it harder? Train a second model with two `
    + `labels removed from its data entirely, then ask it for them at every setting. If guidance `
    + `conjured concepts, turning it up would eventually produce something.`);

  set('live-intro',
    `Pick a digit and turn the dial. Every step costs two passes through the score network rather `
    + `than one, because guidance needs the answer with the label and the answer without it, and `
    + `steps along the difference between them. That is the price of the dial and it is paid at `
    + `sampling time, not at training time.`);

  set('verdict-intro',
    `Three questions, and the third is the one that is easy to get wrong.`);

  set('verdict-obey',
    `Guidance works, and fast: ${pc(zero.obeys)} of requests answered correctly at zero, `
    + `${pc(bestObey.obeys)} at ${bestObey.scale}.`);
  set('verdict-obey-warn',
    `${bestObey.obeys > data.real.obeys
      ? `Past a point it obeys better than real digits do, which means it has stopped drawing `
        + `digits and started drawing the average of each one.`
      : `Obedience never passes what real digits score, so nothing here is over-obeying.`}`);

  set('verdict-real',
    `Best at ${bestReal.scale} (${bestReal.mmd2}), which is `
    + `${split ? 'not where obedience is best' : 'also where obedience is best'}.`);
  set('verdict-real-warn',
    `Variety falls from ${zero.variety} at zero to ${bestObey.variety} at ${bestObey.scale}, `
    + `against ${data.real.variety} for real digits. Realism and variety are not the same thing `
    + `and this table needs both.`);

  set('verdict-held',
    `${pc(H.unseen)} for labels removed from training against ${pc(H.seen)} for labels kept, `
    + `averaged over the settings measured.`);
  set('verdict-held-warn',
    `${H.unseen < 0.1
      ? 'Below guessing, and it gets worse as the dial goes up. There is no setting that makes a '
        + 'model produce something it was never shown.'
      : 'About what guessing gives, so the dial neither helps nor hurts for an unknown label.'}`);

  set('closing-1',
    `The useful shape of this is a rule of thumb with a measurement under it. Conditioning is `
    + `nearly free to train, guidance is a dial you can turn after training, and the setting you `
    + `want is not the one that maximises obedience. `
    + `${split
      ? `Here it is ${bestReal.scale} rather than ${bestObey.scale}, and the gap between them is `
        + `visible in the pictures long before it is visible in the obedience number.`
      : `Here both agree at ${bestObey.scale}, which is a comfortable case and not the usual one.`} `
    + `The failure past that point has a name a reader can carry: the model stops drawing digits `
    + `and starts drawing the idea of each digit.`);

  set('closing-2',
    `What is left in this module is the cost that has been in the background of every article `
    + `since the first one. Sampling takes a chain, the chain takes steps, and each step is a `
    + `pass through a network. <a href="../ddpm/">The pixel article</a> measured how many of those `
    + `steps its samples actually needed, and this one doubles the cost of every single one of `
    + `them, because guidance runs the network twice. The last article of this module asks `
    + `whether that chain can be collapsed to a handful of steps, or to one, and what is lost `
    + `when it is.`
    + `<br /><br />`
    + `The article after this one in the atlas changes the object rather than the method. `
    + `Everything so far has started from a row of numbers or a picture, and both arrive already `
    + `numeric. A <span class="bold">document</span> does not: it is a sequence of words of no `
    + `fixed length, from a vocabulary nobody wrote down, and something has to decide what its `
    + `numbers are before any of this machinery can touch it. `
    + `<a href="../tf-idf/">Counting words</a> is where that starts.`);

  set('ref-note',
    `The autoencoder and the digits are the previous article's, checked by reproducing its `
    + `published round trip. The conditional model is ${M.hidden} wide and ${M.depth} deep on a `
    + `${M.k} number latent, trained for ${n0(M.epochs)} epochs with the label hidden `
    + `${pc(M.drop, 0)} of the time, which is what leaves one set of weights able to answer both `
    + `with and without it. Sampling uses ${M.diff_steps} noise levels on a cosine schedule. `
    + `Obedience is the shared judge's argmax over ${n0(M.asked)} requests with labels drawn `
    + `uniformly; the judge is ${(M.judge_accuracy * 100).toFixed(2)}% accurate on real held out `
    + `digits and that number is on the chart as the line to read the others against. Variety is `
    + `the mean euclidean distance between two draws of the same requested label. The held out `
    + `experiment is a second model trained on ${n0(H.n_train)} digits with labels `
    + `${H.labels.join(' and ')} removed. The weights the page runs are ${n0(data.net.count)} `
    + `numbers in float16, checked against a ${data.net.probe.z.length} point probe covering both `
    + `the score network and the decoder before anything is drawn.`);

  return IDS;
}
