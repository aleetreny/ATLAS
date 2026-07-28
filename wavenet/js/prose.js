/* Every figure on this page, composed from the file it was measured into. */
const n0 = (v) => v.toLocaleString('en-US');
const pc = (v, d = 1) => `${(v * 100).toFixed(d)}%`;

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

export function initProse(data) {
  const M = data.meta;
  const Q = data.quantise;
  const R = data.receptive;
  const F = data.free;
  const A = data.recognition;
  const lin = Q.rows.find((r) => r.scheme.startsWith('linear, 8'));
  const law = Q.rows.find((r) => r.scheme.startsWith('mu-law, 8'));
  const linQ = Q.rows.find((r) => r.scheme.includes('quiet') && r.scheme.startsWith('linear'));
  const lawQ = Q.rows.find((r) => r.scheme.includes('quiet') && r.scheme.startsWith('mu-law'));
  const rows = R.rows;
  const under = rows.filter((d) => !d.covers_period);
  const over = rows.filter((d) => d.covers_period);
  const biggest = rows.slice(1).reduce((best, row, i) => {
    const jump = rows[i].nll - row.nll;
    return jump > best.jump ? { jump, from: rows[i], to: row } : best;
  }, { jump: -Infinity, from: rows[0], to: rows[1] });
  const freeArms = ['0', '0.7', '1'].map((k) => ({ k, ...F.arms[k] }));
  const bestFree = freeArms.reduce((a, b) => (b.spec_distance < a.spec_distance ? b : a),
    freeArms[0]);

  set('opening-note',
    `Both models on this page are trained here, from scratch, on sound made by the same source and `
    + `filter as <a href="../spectrograms/">the previous article</a>: `
    + `${Object.keys(M.vowels).length} vowels and three fricatives with written down resonances, `
    + `at ${n0(M.sr)} samples a second. That matters more for the second half than the first. A `
    + `recogniser can only be scored on its transcript unless somebody has marked where each sound `
    + `starts, and nobody can mark that to the sample; here the boundaries were decided before the `
    + `sound existed, so the alignment can be scored in milliseconds.`);

  set('mu-intro',
    `Start with a decision that looks like an implementation detail and is not. A model that `
    + `generates audio one sample at a time has to choose a value, and choosing among `
    + `${M.levels} categories is far easier to train than regressing a real number. But sixteen `
    + `bit audio has ${n0(65536)} levels, so ${M.levels} of them have to be placed somewhere, and `
    + `where is the question the mu-law curve answers.`);

  set('mu-note',
    `Read the two columns in that order. Over the whole clip the difference is `
    + `${Math.abs(law.snr_db - lin.snr_db).toFixed(2)} dB in favour of `
    + `${law.snr_db > lin.snr_db ? 'mu-law' : 'the even spacing'}; over the quiet `
    + `${pc(Q.quiet_share, 0)} of samples it is `
    + `${Math.abs(lawQ.snr_db - linQ.snr_db).toFixed(2)} dB in favour of `
    + `${lawQ.snr_db > linQ.snr_db ? 'mu-law' : 'the even spacing'}. `
    + `${lawQ.snr_db > linQ.snr_db && law.snr_db <= lin.snr_db
      ? `Those two point different ways, and that is the whole argument for a compander: a signal `
        + `to noise ratio computed over everything is dominated by the loudest samples, which is `
        + `precisely where even spacing is at its best and where the ear is least sensitive to `
        + `the difference.`
      : `The quiet samples are where the compander earns its place, and they are most of what a `
        + `listener spends time on.`}`);

  set('scrolly-intro',
    `Now the model. A convolution that is allowed to look forward is useless for generation, so `
    + `every layer is shifted to see only the past. Stacked, that gives a reach of one sample per `
    + `layer, which for a second of audio would need thousands of layers. The fix is to skip: `
    + `layer k looks at the sample 2<sup>k</sup> steps back rather than the one immediately behind, `
    + `and the reach doubles with every layer instead of adding one.`);

  set('scrolly-step-0',
    `One layer, kernel of two, shifted: the output at each moment sees this sample and the one `
    + `before it. A reach of 2 samples, which at ${n0(M.sr)} samples a second is `
    + `${(2000 / M.sr).toFixed(2)} milliseconds.`);

  set('scrolly-step-1',
    `The second layer takes its two inputs two apart instead of one. Nothing is skipped in the `
    + `output, because the layer below has already looked at every sample; what changes is how far `
    + `the pair reaches.`);

  set('scrolly-step-2',
    `Keep doubling. With dilations 1, 2, 4, 8, 16 the reach is `
    + `1 + 1 + 2 + 4 + 8 + 16 = 32 samples, and the count is exact: the receptive field of a stack `
    + `of dilated causal convolutions with kernel two is one plus the sum of its dilations. No `
    + `training, no data, no tuning. Arithmetic.`);

  set('scrolly-step-3',
    `So here is a prediction. The sound these models are trained on is a vowel at ${R.f0} hertz, `
    + `whose period is <span class="bold">${R.period} samples</span> (${R.period_ms} `
    + `milliseconds). A stack that cannot see one whole period cannot know where in the period it `
    + `is, and the largest improvement in the table should therefore be the step that crosses `
    + `${R.period}. Five stacks, ${rows[0].layers} to ${rows[rows.length - 1].layers} layers, `
    + `trained identically on the same ${R.seconds} seconds.`);

  const crossing = rows.slice(1).map((row, i) => ({
    from: rows[i], to: row, drop: rows[i].nll - row.nll,
  })).find((step) => step.from.receptive < R.period && step.to.receptive >= R.period);
  set('dilation-note',
    `${!crossing
      ? `Every stack here is on the same side of a period, so the crossing is not on this chart.`
      : crossing.from === biggest.from
        ? `The prediction holds: the largest single improvement in held out loss is exactly the `
          + `step that crosses a period, from ${crossing.from.receptive} samples to `
          + `${crossing.to.receptive} (${crossing.from.nll} to ${crossing.to.nll}).`
        : `<span class="bold">The prediction is only half right, and the half that fails is worth `
          + `more than the half that works.</span> The step that crosses a period does improve, `
          + `from ${crossing.from.receptive} samples to ${crossing.to.receptive} `
          + `(${crossing.from.nll} to ${crossing.to.nll}, a drop of `
          + `${crossing.drop.toFixed(4)}), but it is not the largest step in the table: that is `
          + `${biggest.from.receptive} to ${biggest.to.receptive}, a drop of `
          + `${biggest.jump.toFixed(4)}. A reach of ${biggest.to.receptive} samples is `
          + `${(biggest.to.receptive / R.period).toFixed(2)} of a period, and most of what there `
          + `is to learn about a waveform is local: the shape of one glottal pulse, not where the `
          + `next one falls. Covering a whole period buys the last of it, not the bulk.`} `
    + `The bars below are the same five models scored differently, as the signal to noise ratio of `
    + `the waveform they predict, which is the number a listener would care about.`);

  set('receptive-note',
    `Two things worth taking from that. First, the depth needed is a division and not a search: `
    + `for speech at ${n0(M.sr)} samples a second you need `
    + `${Math.ceil(Math.log2(R.period))} layers to hear one pitch period, and about `
    + `${Math.ceil(Math.log2(0.1 * M.sr))} to hear a tenth of a second of context, which is why `
    + `real WaveNets stack the pattern several times over. Second, the cost of the last doubling `
    + `is not the parameters (${rows[rows.length - 1].params.toLocaleString('en-US')} against `
    + `${rows[0].params.toLocaleString('en-US')}, a factor of `
    + `${(rows[rows.length - 1].params / rows[0].params).toFixed(1)}) but the arithmetic per `
    + `sample, and a sample is produced ${n0(M.sr)} times a second.`);

  set('free-intro',
    `Everything above was measured with the model looking at real samples. That is not how it is `
    + `used. To generate, it is given its own previous outputs, and those are not real samples: `
    + `they are its best guesses, complete with whatever it got slightly wrong. After `
    + `${n0(F.generated)} steps of that, is it still making the sound it was trained on?`);

  set('free-note',
    `The honest summary is that a small model run free drifts, and how it drifts depends on how it `
    + `is sampled. The best of the three arms here is `
    + `${bestFree.k === '0' ? 'always taking the likeliest sample'
      : `sampling at temperature ${bestFree.k}`} at ${bestFree.spec_distance} of spectral `
    + `distance. This is the gap that scheduled sampling, and later the whole family of `
    + `non-autoregressive vocoders, was invented to close, and it is not a small model's `
    + `peculiarity: it is a property of training on the truth and running on yourself.`);

  set('asr-intro',
    `The other half of the field goes the other way. It never touches a waveform: it takes the log `
    + `mel picture from the previous article, ${M.n_mels} numbers `
    + `${A.frames_per_second} times a second, and produces symbols. Two decoders on exactly the `
    + `same features and the same ${n0(A.n_train)} utterances: one trained with a loss that sums `
    + `over every possible alignment, and one trained to emit symbols in order, attending over the `
    + `encoder, like Whisper's.`);

  set('asr-note',
    `Both learn it. What separates them is what they do at the edges, and the second view of the `
    + `widget is there to make the point that neither of them sees anything else: `
    + `${M.n_mels} numbers per frame is the whole input, and the waveform, the pitch and the `
    + `phase are all already gone.`);

  const at = document.querySelector('#asr-table');
  if (at) {
    at.innerHTML = `
      <tr>
        <td><span class="bold">alignment free (CTC)</span></td>
        <td><span class="value">${pc(A.ctc.ser)}</span></td>
        <td>0 by construction</td>
        <td>0 by construction</td>
      </tr>
      <tr>
        <td><span class="bold">attention decoder</span></td>
        <td><span class="value">${pc(A.decoder.ser)}</span></td>
        <td>${A.decoder.too_long} of ${A.n_test}</td>
        <td>${A.decoder.runaway} of ${A.n_test}</td>
      </tr>`;
  }

  set('asr-table-note',
    `The first head cannot produce an output longer than the number of frames, and it cannot `
    + `repeat a symbol unless the sound repeats, because repeats have to be separated by a blank. `
    + `The second can do anything: it decides for itself when to emit an end symbol, and `
    + `${A.decoder.too_long > 0
      ? `on ${A.decoder.too_long} of ${A.n_test} test utterances it decided wrong, `
        + `${A.decoder.runaway} of them by looping on one symbol. That is the failure everybody `
        + `who has run a large speech model has seen: a transcript that keeps going after the `
        + `audio has stopped, repeating a phrase, because nothing in the architecture ties the `
        + `length of the output to the length of the input.`
      : `on this corpus it always decided right, which is worth reporting as the negative result `
        + `it is: a failure that does not reproduce at this scale.`} `
    + `And the alignment: the spikes of the first head land ${A.alignment.median_ms} milliseconds `
    + `from the middle of their phone in the median, over ${A.alignment.n} phones, against a frame `
    + `step of ${A.alignment.frame_ms} milliseconds. Timestamps came free with a model nobody ever `
    + `told where anything was.`);

  set('verdict-intro', `Four things you might want, and what this page measured.`);
  set('verdict-gen',
    `The reach is one plus the sum of the dilations, so ${rows[rows.length - 1].layers} layers see `
    + `${rows[rows.length - 1].receptive} samples (${rows[rows.length - 1].receptive_ms} ms) and `
    + `score ${rows[rows.length - 1].snr_db} dB against ${rows[0].snr_db} at `
    + `${rows[0].receptive}.`);
  set('verdict-gen-warn',
    `Teacher forcing flatters it: run free for ${n0(F.generated)} samples and the best arm is `
    + `${bestFree.spec_distance} away from the real spectrum. And the compander is not free either `
    + `(${law.snr_db} dB at ${M.levels} levels).`);
  set('verdict-asr',
    `${pc(A.ctc.ser)} symbol error for the alignment free head and ${pc(A.decoder.ser)} for the `
    + `attention decoder, on identical features.`);
  set('verdict-asr-warn',
    `The decoder's length is its own decision: ${A.decoder.too_long} of ${A.n_test} outputs were `
    + `longer than the truth.`);
  set('verdict-align',
    `Spikes land ${A.alignment.median_ms} ms from the middle of their phone, with a frame step of `
    + `${A.alignment.frame_ms} ms.`);
  set('verdict-align-warn',
    `An attention decoder gives you no alignment at all, and the attention weights are not one: `
    + `<a href="../deep-forecast/">the fourth article of the previous branch</a> measured what `
    + `those weights are worth.`);
  set('verdict-trust',
    `A model whose output length is tied to the input cannot run away. That is a property of the `
    + `loss, not of the size of the model.`);
  set('verdict-trust-warn',
    `And it is the property people give up for the accuracy and the language modelling that an `
    + `autoregressive decoder brings.`);

  set('closing-note',
    `That closes the branch, and with it the two ways of handling data that arrives in order. `
    + `Sound turned out to need the same three questions as a series of monthly measurements, `
    + `asked eight thousand times faster: what does the past determine, what is the right `
    + `representation to ask in, and how do you know your evaluation is not measuring itself. `
    + `What is left of the map is what happens when the thing arriving in order is a `
    + `<span class="bold">decision</span> rather than an observation, and the next value depends `
    + `on what you do about it. That is <a href="../">the rest of the atlas</a>, and the module `
    + `after this one is where it starts.`);

  set('resources-note',
    `Audio is synthesised at ${n0(M.sr)} samples a second from the source and filter of the `
    + `previous article, seed ${M.seed}. The generative stacks are ${rows[0].layers} to `
    + `${rows[rows.length - 1].layers} dilated causal layers of 32 channels with gated activations `
    + `and skip connections, over ${M.levels} mu-law levels, trained on ${R.seconds} seconds and `
    + `scored on the last fifth of it. The recognisers share a two layer bidirectional recurrent `
    + `encoder over ${M.n_mels} log mel bands (window ${M.win}, hop ${M.hop}, `
    + `${A.frames_per_second} frames a second) and differ only in the head: a `
    + `${A.symbols.length} symbol alignment free loss, and an attention decoder with a learned end `
    + `symbol and a hard cap of ${A.decoder.cap}. ${n0(A.n_train)} utterances for training and `
    + `${A.n_test} held out. Trained on ${M.threads} threads with torch ${M.torch}.`);
}
