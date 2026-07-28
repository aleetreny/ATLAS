/* Every figure on this page, composed from the file it was measured into. */
const n0 = (v) => v.toLocaleString('en-US');
const pc = (v, d = 0) => `${(v * 100).toFixed(d)}%`;

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

export function initProse(data) {
  const M = data.meta;
  const R = data.resolution;
  const C = data.chirp;
  const I = data.inversion;
  const L = data.mel;
  const P = data.cepstrum;
  const D = data.identify;
  const vib = C.vibrato;
  const bestChirp = C.rows.reduce((a, b) => (b.rms_hz < a.rms_hz ? b : a), C.rows[0]);
  const bestVib = vib.rows.reduce((a, b) => (b.rms_hz < a.rms_hz ? b : a), vib.rows[0]);
  const worstVib = vib.rows[vib.rows.length - 1];
  const mfccRow = D.rows[0];
  const specRow = D.rows[D.rows.length - 1];
  const cutBest = D.cut.reduce((a, b) => (b.accuracy > a.accuracy ? b : a), D.cut[0]);
  const cutFirst = D.cut.find((c) => c.accuracy >= cutBest.accuracy - 1e-9);

  set('opening-note',
    `Every sound on this page is synthetic, and that is the point rather than a shortcut. It is `
    + `made by a source and a filter whose parameters are written down before the first sample `
    + `exists: ${Object.keys(M.vowels).length} vowels with the formant frequencies Peterson and `
    + `Barney measured, three fricatives, and a pitch contour known at every instant. So when the `
    + `page asks what a transform kept and what it threw away, the answer is arithmetic against a `
    + `known quantity rather than a judgement against an annotation. A recorded corpus cannot `
    + `answer any of the questions below, because nobody can mark the boundary between two sounds `
    + `to the sample, and nobody can tell you what the pitch was doing at instant 4,137.`);

  set('scrolly-intro',
    `The transform is four steps and no cleverness at all: take a slice, fade its edges, ask what `
    + `frequencies are in it, slide along and repeat. Everything below is computed in your browser `
    + `from ${n0(I.n)} samples the generator sent, at ${n0(M.sr)} samples a second.`);

  set('scrolly-step-0',
    `A waveform, and the phone boundaries this utterance was built from. It is `
    + `${(I.n / M.sr).toFixed(2)} seconds of ${I.marks.map((m) => m.sym).join(' ')}, and nothing in `
    + `the shape of it says which is which: the information is in how fast it wiggles, not in how `
    + `high it goes.`);

  set('scrolly-step-1',
    `Take ${I.win} samples, ${(1000 * I.win / M.sr).toFixed(0)} milliseconds. The taper matters `
    + `more than it looks: a rectangular slice has two sharp edges, and a sharp edge is a click, `
    + `and a click is energy at every frequency at once. Fading in and out removes an artefact `
    + `that would otherwise appear in the picture and never happened in the sound.`);

  set('scrolly-step-2',
    `The magnitude of the Fourier transform of that slice is one column of the picture. The peaks `
    + `line up with the resonances the vowel was built with, marked here, and the fine comb `
    + `between them is the pitch: two entirely different things in one plot, which is the reason `
    + `the last section of this article exists.`);

  set('scrolly-step-3',
    `Then slide by ${I.hop} samples and do it again, ${n0(Math.floor((I.n - I.win) / I.hop))} `
    + `times. Standing the columns side by side is the spectrogram, and from here on nothing in `
    + `this article touches the waveform again.`);

  set('pipeline-note',
    `Two numbers were chosen in there without comment, and one of them decides what the picture `
    + `can possibly show. The window length sets the frequency spacing to about `
    + `${(M.sr / I.win).toFixed(1)} hertz and the time resolution to `
    + `${(1000 * I.win / M.sr).toFixed(0)} milliseconds, and those two move in opposite `
    + `directions. That is not a limitation of the software.`);

  set('window-intro',
    `A longer window sees finer frequency detail and a shorter one sees finer timing, and the `
    + `product of the two is bounded below no matter what you do. The three sounds below have `
    + `known answers, so the tradeoff can be measured rather than described.`);

  set('window-equation-note',
    `The slider goes from ${C.rows[0].win} samples (${C.rows[0].ms} milliseconds) to `
    + `${C.rows[C.rows.length - 1].win} (${C.rows[C.rows.length - 1].ms} milliseconds). Watch the `
    + `picture, and watch the tracking error underneath it: the accent line is the loudest bin in `
    + `each column and the blue line is the frequency the sound actually had at that instant.`);

  set('window-note',
    `The straight sweep is the case that misleads, and it is worth being precise about why. Its `
    + `error falls all the way to the longest window here (${bestChirp.rms_hz} hertz at `
    + `${bestChirp.win} samples) because a linear sweep smears symmetrically about the middle of `
    + `a window: the bias cancels and only the noise is left to average away. Bend the frequency `
    + `and it stops cancelling. On the tone that wobbles ${vib.rate_hz} times a second, the same `
    + `sweep of window lengths is U shaped, ${bestVib.rms_hz} hertz at its best `
    + `(${bestVib.win} samples, ${bestVib.ms} milliseconds) and ${worstVib.rms_hz} hertz at `
    + `${worstVib.win}, which is `
    + `${(worstVib.rms_hz / bestVib.rms_hz).toFixed(0)} times worse for a window that contains `
    + `${(worstVib.ms * vib.rate_hz / 1000).toFixed(1)} whole cycles of the thing it is trying to `
    + `follow. <span class="bold">The tradeoff is only visible on a signal that moves</span>, `
    + `which is most of them.`);

  const rt = document.querySelector('#resolution-table');
  if (rt) {
    rt.innerHTML = R.rows.map((row) => `<tr>
      <td><span class="bold">${row.sep} hertz</span></td>
      <td>${row.predicted} samples</td>
      <td><span class="value">${row.needed === null ? 'none of them' : `${row.needed} samples`}</span></td>
      <td>${row.needed === null ? '' : (row.needed / row.predicted).toFixed(2)}</td>
    </tr>`).join('');
  }

  set('resolution-note',
    `Two sine waves that close together are one bump until the window is long enough, and how `
    + `long is not a matter of taste: a window of N samples spaces its frequency bins `
    + `${M.sr}/N hertz apart, so telling apart two tones deltaf apart needs N above about `
    + `${M.sr}/deltaf. That is the middle column, computed before any signal was made. The third `
    + `is where the measurement actually crossed over on a grid of powers of two, and the ratio is `
    + `between ${Math.min(...R.rows.filter((x) => x.needed).map((x) => x.needed / x.predicted)).toFixed(2)} `
    + `and ${Math.max(...R.rows.filter((x) => x.needed).map((x) => x.needed / x.predicted)).toFixed(2)}: `
    + `a tapered window widens the main lobe by about a factor of two, which is exactly what it `
    + `pays for the freedom from leakage that the previous step bought.`);

  set('invert-intro',
    `A spectrogram keeps the size of each frequency in each slice and throws away where it was in `
    + `its cycle. That sounds like a detail and it is half the numbers. Here is the same utterance `
    + `three ways: inverted with its phase, reconstructed from the magnitudes alone by `
    + `${I.iters} rounds of Griffin-Lim, and reconstructed from the same magnitudes with the phase `
    + `replaced by noise.`);

  set('invert-note',
    `The exact inversion returns the samples to ${I.exact_error.toExponential(1)}, which is `
    + `machine precision: nothing was lost, because nothing was discarded. The other two are the `
    + `interesting pair, and they are the reason this section is here. Judged by waveform error, `
    + `the guessed phase scores ${I.griffin_snr_db} dB and pure noise scores `
    + `${I.random_phase_snr_db} dB, so the ruler prefers the noise, and both are below zero, which `
    + `would mean worse than silence. Judged by the distance between the two <i>pictures</i>, the `
    + `iterated one is at ${I.spec_gap} and the noise at ${I.spec_gap_random}, `
    + `${(I.spec_gap_random / I.spec_gap).toFixed(1)} times further. Anything trained on `
    + `spectrograms and evaluated on waveforms is choosing the first ruler by accident.`);

  set('mfcc-intro',
    `The last step before a model is almost never the raw spectrogram. It is `
    + `${L.n_mels} numbers per frame instead of ${L.n_bins}, from triangular filters spaced evenly `
    + `on a scale that is linear below 700 hertz and logarithmic above it, then the logarithm, `
    + `then a cosine transform of that. What survives, and what is that last step for?`);

  set('mfcc-note',
    `That last step is a change of coordinates on the log spectrum, and the coordinates are `
    + `chosen so that <span class="bold">how fast the spectrum wiggles</span> becomes an axis. The `
    + `resonances of the mouth are a slow wiggle and land in the first few coefficients; the pitch `
    + `is a fast comb and lands in the high ones, which get thrown away. Measured over `
    + `${P.pitches.length} pitches and ${Object.keys(M.vowels).length} vowels, the ratio of `
    + `between-vowel distance to across-pitch spread is ${P.spec_ratio} on the raw log spectrum `
    + `and ${P.coef_ratio} on the coefficients. And the pitch is not destroyed by the change of `
    + `coordinates, only moved: reading the same cepstrum at high quefrencies recovers it with a `
    + `median error of ${P.pitch.median_error_hz} hertz, with `
    + `${P.pitch.octave_errors} of ${P.pitch.n} coming back an octave low, which is the classic `
    + `failure of that estimator and is reproduced here rather than hidden. `
    + `Rebuilding the ${L.n_bins} bin log spectrum from the ${L.n_mels} mel energies by least `
    + `squares leaves ${pc(L.rebuild_rel_error, 1)} of relative error, so the filterbank is a real `
    + `loss and not a repackaging.`);

  const head = document.querySelector('#identify-head');
  if (head) {
    head.innerHTML = `<th>Features</th>${D.levels.map((l) => `<th>${l}</th>`).join('')}`;
  }
  const it = document.querySelector('#identify-table');
  if (it) {
    it.innerHTML = D.rows.map((row) => `<tr>
      <td><span class="bold">${row.features}</span></td>
      ${D.levels.map((l) => `<td>${pc(row[l], 0)}</td>`).join('')}
    </tr>`).join('');
  }

  set('identify-note',
    `Naming the ${D.n_vowels} vowels from those numbers, trained at pitches `
    + `${D.train_pitches.join(', ')} and tested at ${D.test_pitches.join(', ')} hertz, which the `
    + `classifier has never heard. Clean, everything works and the table says nothing: `
    + `${pc(mfccRow.clean, 0)} for the coefficients and ${pc(specRow.clean, 0)} for the whole `
    + `spectrum. Add noise and it separates. At 25 decibels the coefficients hold `
    + `${pc(mfccRow['25 dB'], 0)} while the raw log spectrum has already collapsed to `
    + `${pc(specRow['25 dB'], 0)}, which is chance for ${D.n_vowels} classes. That is what the `
    + `filterbank is for: ${L.n_mels} smoothed bands are a far more robust description of a shape `
    + `than ${L.n_bins} exact ones. Below 15 decibels everything here is at chance, and this page `
    + `says so rather than stopping at the level that flatters the method. `
    + `How many coefficients does it take? At ${D.cut_snr} decibels, accuracy reaches `
    + `${pc(cutBest.accuracy, 0)} by ${cutFirst.coefficients} coefficients and does not improve `
    + `after that, which is why the number thirteen has survived since 1980 with nobody quite able `
    + `to justify the last few.`);

  set('verdict-intro', `Four things you might want from a picture of a sound.`);
  set('verdict-long',
    `Bins are ${M.sr}/N hertz apart, so ${R.rows[0].sep} hertz apart needs about `
    + `${R.rows[0].predicted} samples and measured out at ${R.rows[0].needed}.`);
  set('verdict-long-warn',
    `The same window blurs anything that moves: ${worstVib.rms_hz} hertz of tracking error on a `
    + `tone wobbling ${vib.rate_hz} times a second, against ${bestVib.rms_hz} at `
    + `${bestVib.win} samples.`);
  set('verdict-short',
    `A window is a time resolution: ${C.rows[0].ms} milliseconds at ${C.rows[0].win} samples, and `
    + `nothing shorter than that is locatable in the picture.`);
  set('verdict-short-warn',
    `And the bins get correspondingly coarse. Neither end is the right answer; the sound decides.`);
  set('verdict-mel',
    `${L.n_mels} numbers instead of ${L.n_bins}, and ${pc(mfccRow['25 dB'], 0)} against `
    + `${pc(specRow['25 dB'], 0)} at 25 decibels of noise.`);
  set('verdict-mel-warn',
    `It is lossy: ${pc(L.rebuild_rel_error, 1)} of relative error rebuilding the spectrum, and `
    + `the pitch removed on purpose. If the pitch is what you want, the coefficients are the wrong `
    + `features.`);
  set('verdict-phase',
    `With the phase the inversion is exact to ${I.exact_error.toExponential(1)}.`);
  set('verdict-phase-warn',
    `Without it, ${I.iters} iterations get to ${I.spec_gap} in picture distance and `
    + `${I.griffin_snr_db} dB in waveform terms, which is a number that will mislead you if you `
    + `report it.`);

  set('closing-note',
    `Everything here has been a transform: fixed arithmetic, no parameters, no training. The two `
    + `models that made audio interesting again both start where this article stops. One works on `
    + `the waveform itself, one sample at a time, and had to solve the problem that a second of `
    + `sound is ${n0(M.sr)} steps of context. The other never looks at a waveform at all: it takes `
    + `the log mel picture from this page, hands it to a transformer, and writes down words. `
    + `<a href="../wavenet/">The next article</a> builds both, and measures the two things they `
    + `are most often accused of: that the receptive field of a stack of dilated convolutions is `
    + `exactly computable and often too short, and that an autoregressive decoder will happily `
    + `keep talking after the sound has stopped.`);

  set('resources-note',
    `Sounds are synthesised at ${n0(M.sr)} samples a second from a source and a three pole filter: `
    + `vowel formants ${Object.entries(M.vowels).map(([k, v]) => `${k} (${v[0]}, ${v[1]})`).join(', ')} `
    + `hertz with bandwidths ${M.bandwidths.join(', ')}, and fricatives at `
    + `${Object.entries(M.fricatives).map(([k, v]) => `${k} (${v[0]})`).join(', ')}. The transform `
    + `is a periodic Hann window with a ${M.nfft} point transform; the filterbank is ${L.n_mels} `
    + `triangles from 0 to ${M.sr / 2} hertz and the cepstrum keeps ${M.n_mfcc} coefficients. `
    + `Griffin-Lim runs ${I.iters} iterations from a random phase with seed ${M.seed}. Every `
    + `spectrogram on this page is recomputed in the browser and checked against the generator's `
    + `on load. Measured with numpy ${M.numpy} and scipy ${M.scipy}.`);
}
