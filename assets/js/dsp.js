/* ATLAS signal helpers: the transforms module 4.3 runs in the browser.
 *
 * Written out rather than imported so that both sides of every measurement in
 * that module are visible: the generator computes a spectrogram in numpy and
 * the page computes one here, and the guard on each page compares them. Two
 * conventions are pinned down, because they are where two implementations of
 * "the spectrogram" usually differ:
 *
 *   - the window is periodic Hann, w[n] = 0.5 - 0.5*cos(2*pi*n/N), which is
 *     what numpy's `hanning(N+1)[:-1]` and scipy's `get_window` give, and NOT
 *     the symmetric `numpy.hanning(N)`;
 *   - the magnitude is not normalised by anything, so a decibel value here is
 *     20*log10(|X|) with no reference, and only differences are meaningful.
 *
 * No DOM in this file.
 */

/* In place iterative radix-2 FFT. `re` and `im` must have a power of two
 * length. Not the fastest arrangement, but it is fifty lines and it is the
 * same arithmetic as the reference. */
export function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang);
    const wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1;
      let ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const ur = re[i + k];
        const ui = im[i + k];
        const vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
        const vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
        re[i + k] = ur + vr;
        im[i + k] = ui + vi;
        re[i + k + len / 2] = ur - vr;
        im[i + k + len / 2] = ui - vi;
        const ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = ncr;
      }
    }
  }
}

export function nextPow2(n) {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

/* Periodic Hann, the convention every library that does overlap-add uses. */
export function hann(n) {
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n);
  return w;
}

/* Magnitude spectrum of one frame, zero padded to `nfft`. Returns nfft/2 + 1
 * numbers, the same as a real FFT. */
export function magSpectrum(frame, nfft) {
  const re = new Float64Array(nfft);
  const im = new Float64Array(nfft);
  re.set(frame.slice(0, Math.min(frame.length, nfft)));
  fft(re, im);
  const half = nfft / 2 + 1;
  const out = new Float64Array(half);
  for (let k = 0; k < half; k++) out[k] = Math.hypot(re[k], im[k]);
  return out;
}

/* Short time Fourier magnitudes: frames of `win` samples every `hop`.
 * Returns {frames, freqs, times} with frames[t][k]. */
export function stft(x, { win = 256, hop = 64, sr = 8000, nfft = null } = {}) {
  const N = nfft || nextPow2(win);
  const w = hann(win);
  const frames = [];
  const times = [];
  for (let start = 0; start + win <= x.length; start += hop) {
    const frame = new Float64Array(N);
    for (let i = 0; i < win; i++) frame[i] = x[start + i] * w[i];
    frames.push(magSpectrum(frame, N));
    times.push((start + win / 2) / sr);
  }
  const freqs = new Float64Array(N / 2 + 1);
  for (let k = 0; k < freqs.length; k++) freqs[k] = (k * sr) / N;
  return { frames, freqs, times, win, hop, nfft: N };
}

export const hzToMel = (f) => 2595 * Math.log10(1 + f / 700);
export const melToHz = (m) => 700 * (10 ** (m / 2595) - 1);

/* Triangular mel filterbank, as a matrix of `nMels` rows over `nBins` bins. */
export function melBank(nMels, nfft, sr, fmin = 0, fmax = null) {
  const top = fmax || sr / 2;
  const nBins = nfft / 2 + 1;
  const points = [];
  for (let i = 0; i < nMels + 2; i++) {
    points.push(melToHz(hzToMel(fmin) + ((hzToMel(top) - hzToMel(fmin)) * i) / (nMels + 1)));
  }
  const bins = points.map((f) => Math.floor(((nfft + 1) * f) / sr));
  const bank = [];
  for (let m = 1; m <= nMels; m++) {
    const row = new Float64Array(nBins);
    for (let k = bins[m - 1]; k < bins[m]; k++) {
      if (k >= 0 && k < nBins && bins[m] > bins[m - 1]) {
        row[k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1]);
      }
    }
    for (let k = bins[m]; k < bins[m + 1]; k++) {
      if (k >= 0 && k < nBins && bins[m + 1] > bins[m]) {
        row[k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m]);
      }
    }
    bank.push(row);
  }
  return bank;
}

/* Orthonormal DCT-II of a real vector, the one every MFCC uses. */
export function dct2(x) {
  const n = x.length;
  const out = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += x[i] * Math.cos((Math.PI * k * (2 * i + 1)) / (2 * n));
    out[k] = s * (k === 0 ? Math.sqrt(1 / (4 * n)) : Math.sqrt(1 / (2 * n))) * 2;
  }
  return out;
}

/* Log mel energies of one magnitude spectrum, then the cepstrum of those. */
export function mfcc(mag, bank, nCoef = 13, floorDb = 1e-10) {
  const logmel = new Float64Array(bank.length);
  for (let m = 0; m < bank.length; m++) {
    let s = 0;
    const row = bank[m];
    for (let k = 0; k < row.length; k++) s += row[k] * mag[k] * mag[k];
    logmel[m] = Math.log(Math.max(s, floorDb));
  }
  return { logmel, coef: Array.from(dct2(logmel)).slice(0, nCoef) };
}

/* One two-pole resonator, the same difference equation the synthesiser used. */
export function resonator(x, freq, bw, sr = 8000) {
  const r = Math.exp((-Math.PI * bw) / sr);
  const theta = (2 * Math.PI * freq) / sr;
  const a1 = -2 * r * Math.cos(theta);
  const a2 = r * r;
  const gain = 1 - 2 * r * Math.cos(theta) + r * r;
  const y = new Float64Array(x.length);
  for (let n = 0; n < x.length; n++) {
    let v = gain * x[n];
    if (n >= 1) v -= a1 * y[n - 1];
    if (n >= 2) v -= a2 * y[n - 2];
    y[n] = v;
  }
  return y;
}

/* Decode an array of 16 bit samples that arrived as a base64 string. */
export function decodeInt16(b64) {
  const bin = atob(b64);
  const out = new Float64Array(bin.length / 2);
  for (let i = 0; i < out.length; i++) {
    const lo = bin.charCodeAt(2 * i);
    const hi = bin.charCodeAt(2 * i + 1);
    let v = lo | (hi << 8);
    if (v >= 32768) v -= 65536;
    out[i] = v / 32767;
  }
  return out;
}

/* Play a buffer of samples through the Web Audio API. Returns the source node
 * so a caller can stop it. */
export function play(samples, sr = 8000) {
  const Ctx = window.AudioContext || window.webkitAudioContext;
  if (!Ctx) return null;
  const ctx = play.ctx || (play.ctx = new Ctx());
  const buf = ctx.createBuffer(1, samples.length, sr);
  buf.copyToChannel(Float32Array.from(samples), 0);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  src.start();
  return src;
}
