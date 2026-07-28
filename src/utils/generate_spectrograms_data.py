"""Everything article 4.3a cites, measured.

A spectrogram is two choices (how long a window, how much overlap) and every
consequence of those choices is arithmetic, not taste. This file measures the
consequences against sounds whose truth is written down rather than annotated:

 1. Resolution. Two tones a known distance apart are resolved by a window
    longer than about 1/deltaf, and that closed form is put beside the measured
    threshold at six separations. A sweep whose instantaneous frequency is
    known at every sample is tracked at eight window lengths, and the error is
    U shaped with a measurable minimum: too short is noisy, too long is smeared.
 2. What the picture throws away. The transform is inverted exactly (to 1e-15)
    when the phase is kept, and by Griffin-Lim when it is not, and the second
    is scored in decibels rather than described.
 3. The mel filterbank as a matrix: 257 numbers into 26, and the reconstruction
    error that costs.
 4. What a cepstral coefficient is for. The same vowel is synthesised at eleven
    pitches, so the filter is identical and the source is not: the spread of
    the log spectrum and the spread of the first thirteen MFCCs answer, in one
    number each, what "MFCCs discard the pitch" means. Then the pitch is read
    back out of the high quefrencies and compared with the pitch that was
    written down.

Runs in about three minutes. Stages cached under
~/.atlas_vision_data/spectrograms_cache/.
"""
import base64
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "spectrograms" / "data"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path.home() / ".atlas_vision_data" / "spectrograms_cache"
CACHE.mkdir(parents=True, exist_ok=True)

from src.utils.audio_data import (  # noqa: E402
    BANDWIDTHS, FRICATIVES, SR, VOWELS, chirp, envelope_db, two_tones,
    to_int16, utterance, vowel,
)

SEED = 46
NFFT = 512
N_MELS = 26
N_MFCC = 13


def cached(name, fn):
    f = CACHE / f"{name}.json"
    if f.is_file():
        return json.loads(f.read_text())
    t0 = time.time()
    out = fn()
    f.write_text(json.dumps(out))
    print(f"  [{name}] {time.time() - t0:.1f}s")
    return out


def r(x, k=4):
    if isinstance(x, (list, tuple, np.ndarray)):
        return [r(v, k) for v in np.asarray(x, dtype=float).tolist()]
    return round(float(x), k)


def b64(x):
    return base64.b64encode(to_int16(x).tobytes()).decode("ascii")


def as_sent(x):
    """What the browser will actually have: sixteen bit samples, decoded back.

    Any reference spectrum computed from the float signal instead is a
    reference for a different waveform. On this page that mattered by 38 dB,
    because the bins far from the sweep sit below the quantisation floor and are
    pure rounding noise once the audio has been through int16.
    """
    return to_int16(x).astype(np.float64) / 32767.0


# ---------------------------------------------------------------------------
# The transform, written out so the browser can be checked against it.
# ---------------------------------------------------------------------------
def hann(n):
    """Periodic, which is what every overlap-add implementation uses and what
    `numpy.hanning(n)` is not."""
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def stft(x, win=256, hop=64, nfft=None):
    nfft = nfft or int(2 ** np.ceil(np.log2(win)))
    w = hann(win)
    starts = range(0, len(x) - win + 1, hop)
    frames = np.array([np.fft.rfft(x[s:s + win] * w, nfft) for s in starts])
    times = np.array([(s + win / 2) / SR for s in starts])
    freqs = np.fft.rfftfreq(nfft, 1 / SR)
    return frames, freqs, times


def istft(frames, win, hop, length):
    """Overlap add with the same window twice, divided by the window sum, which
    is exact for a Hann window at 75% overlap."""
    w = hann(win)
    out = np.zeros(length)
    norm = np.zeros(length)
    for i in range(frames.shape[0]):
        seg = np.fft.irfft(frames[i])[:win]
        s = i * hop
        out[s:s + win] += seg * w
        norm[s:s + win] += w * w
    keep = norm > 1e-8
    out[keep] /= norm[keep]
    return out


def griffin_lim(mag, win, hop, length, iters=60, seed=SEED):
    rng = np.random.default_rng(seed)
    angle = np.exp(2j * np.pi * rng.random(mag.shape))
    x = istft(mag * angle, win, hop, length)
    for _ in range(iters):
        f, _, _ = stft(x, win, hop)
        f = f[:mag.shape[0]]
        angle = np.exp(1j * np.angle(f))
        x = istft(mag[:f.shape[0]] * angle, win, hop, length)
    return x


def mel_bank(n_mels=N_MELS, nfft=NFFT, sr=SR, fmin=0.0, fmax=None):
    top = fmax or sr / 2

    def to_mel(f):
        return 2595 * np.log10(1 + f / 700)

    def to_hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    pts = to_hz(np.linspace(to_mel(fmin), to_mel(top), n_mels + 2))
    bins = np.floor((nfft + 1) * pts / sr).astype(int)
    bank = np.zeros((n_mels, nfft // 2 + 1))
    for m in range(1, n_mels + 1):
        lo, mid, hi = bins[m - 1], bins[m], bins[m + 1]
        if mid > lo:
            bank[m - 1, lo:mid] = (np.arange(lo, mid) - lo) / (mid - lo)
        if hi > mid:
            bank[m - 1, mid:hi] = (hi - np.arange(mid, hi)) / (hi - mid)
    return bank


def dct2(x):
    n = len(x)
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    m = np.cos(np.pi * k * (2 * i + 1) / (2 * n))
    scale = np.where(np.arange(n) == 0, np.sqrt(1 / (4 * n)), np.sqrt(1 / (2 * n))) * 2
    return (m @ np.asarray(x)) * scale


def mfcc_of(mag, bank, n_coef=N_MFCC):
    energy = bank @ (mag ** 2)
    logmel = np.log(np.maximum(energy, 1e-10))
    return dct2(logmel)[:n_coef], logmel


# ---------------------------------------------------------------------------
# Stage 1: resolution, against the closed form
# ---------------------------------------------------------------------------
def stage_resolution():
    seps = [20, 40, 60, 100, 160, 250]
    wins = [32, 64, 128, 256, 512, 1024]
    rows = []
    for df in seps:
        x = two_tones(1000, 1000 + df, 0.6)
        found = None
        detail = []
        for win in wins:
            frames, freqs, _ = stft(x, win=win, hop=win // 4, nfft=max(2048, win * 4))
            mag = np.abs(frames).mean(axis=0)
            band = (freqs > 800) & (freqs < 1400)
            m = mag[band]
            f = freqs[band]
            peaks = [i for i in range(1, len(m) - 1)
                     if m[i] > m[i - 1] and m[i] > m[i + 1] and m[i] > 0.25 * m.max()]
            two = len(peaks) >= 2 and (f[peaks[-1]] - f[peaks[0]]) > df * 0.5
            detail.append({"win": win, "peaks": len(peaks), "resolved": bool(two)})
            if two and found is None:
                found = win
        rows.append({"sep": df, "needed": found, "predicted": int(np.ceil(SR / df)),
                     "detail": detail})
    return {"rows": rows, "wins": wins, "sr": SR,
            "example": {"sep": 40, "win_short": 64, "win_long": 512}}


def stage_chirp():
    x, f_true = chirp(300, 3200, 0.5)
    rows = []
    for win in [16, 32, 64, 128, 256, 512, 1024, 2048]:
        hop = max(1, win // 8)
        frames, freqs, times = stft(x, win=win, hop=hop, nfft=max(1024, win * 2))
        mag = np.abs(frames)
        peak = freqs[np.argmax(mag, axis=1)]
        idx = np.clip((times * SR).astype(int), 0, len(f_true) - 1)
        err = np.sqrt(np.mean((peak - f_true[idx]) ** 2))
        rows.append({"win": win, "ms": r(1000 * win / SR, 2),
                     "rms_hz": r(float(err), 3), "frames": int(len(times)),
                     "bin_hz": r(SR / max(1024, win * 2), 3),
                     "sweep_per_window": r(2900 * (win / SR) / 0.5, 2)})
    best = min(rows, key=lambda z: z["rms_hz"])

    # A linear sweep is the friendliest case there is: over a long window the
    # smearing is symmetric about the middle, so the peak stays at the
    # instantaneous frequency of the window's centre and a longer window simply
    # averages away more noise. Bend the frequency instead and the bias no
    # longer cancels, which is where the tradeoff everyone draws actually lives.
    t = np.arange(int(0.5 * SR)) / SR
    f_vib = 1200 + 300 * np.sin(2 * np.pi * 8 * t)
    x_v = np.sin(2 * np.pi * np.cumsum(f_vib) / SR)
    vib = []
    for win in [16, 32, 64, 128, 256, 512, 1024, 2048]:
        hop = max(1, win // 8)
        frames, freqs, times = stft(x_v, win=win, hop=hop, nfft=max(1024, win * 2))
        peak = freqs[np.argmax(np.abs(frames), axis=1)]
        idx = np.clip((times * SR).astype(int), 0, len(f_vib) - 1)
        vib.append({"win": win, "ms": r(1000 * win / SR, 2),
                    "rms_hz": r(float(np.sqrt(np.mean((peak - f_vib[idx]) ** 2))), 3)})
    best_vib = min(vib, key=lambda z: z["rms_hz"])

    sent = as_sent(x * 0.6)
    frames, freqs, times = stft(sent, win=256, hop=32, nfft=512)
    return {"rows": rows, "best": best["win"], "best_rms": best["rms_hz"],
            "vibrato": {"rows": vib, "best": best_vib["win"], "best_rms": best_vib["rms_hz"],
                        "rate_hz": 8, "depth_hz": 300, "centre_hz": 1200,
                        "audio": b64(x_v * 0.6),
                        "truth": r(f_vib[::40], 2)},
            "duration": 0.5, "f_start": 300, "f_end": 3200,
            "audio": b64(x * 0.6), "audio_peak": r(float(np.max(np.abs(sent))), 4),
            "spec": {"win": 256, "hop": 32, "nfft": 512,
                     "db": r(20 * np.log10(np.abs(frames[::2, :129]) + 1e-8), 2),
                     "times": r(times[::2], 4), "freqs": r(freqs[:129], 2)},
            "truth": r(f_true[::40], 2)}


# ---------------------------------------------------------------------------
# Stage 2: what the picture keeps and what it throws away
# ---------------------------------------------------------------------------
def stage_inversion():
    y, marks = utterance(list("aSieo"), seed=SEED)
    win, hop = 256, 64
    frames, _, _ = stft(y, win, hop)
    exact = istft(frames, win, hop, len(y))
    n = min(len(exact), len(y))
    valid = slice(win, n - win)
    err_exact = float(np.max(np.abs(exact[valid] - y[valid])))
    mag = np.abs(frames)
    gl = griffin_lim(mag, win, hop, len(y), iters=60)
    m = min(len(gl), len(y))
    num = float(np.sum(y[win:m - win] ** 2))
    den = float(np.sum((y[win:m - win] - gl[win:m - win]) ** 2))
    snr = 10 * np.log10(num / max(den, 1e-30))
    # the same magnitudes with random phase, not iterated at all
    rng = np.random.default_rng(SEED)
    rand = istft(mag * np.exp(2j * np.pi * rng.random(mag.shape)), win, hop, len(y))
    den_r = float(np.sum((y[win:m - win] - rand[win:m - win]) ** 2))
    snr_rand = 10 * np.log10(num / max(den_r, 1e-30))

    # Waveform error is the wrong ruler for a phase problem and this is where
    # that shows: both reconstructions come out with a NEGATIVE signal to noise
    # ratio, which would say they are worse than silence, while one of them is
    # a recognisable utterance and the other is a wash. The ruler that matches
    # what was preserved is the distance between the two pictures.
    def spec_gap_of(sig):
        f, _, _ = stft(sig, win, hop)
        return float(np.mean(np.abs(np.abs(f[:frames.shape[0]]) - mag)) / np.mean(mag))

    spec_gap = spec_gap_of(gl)
    spec_gap_rand = spec_gap_of(rand)
    spec_gap_silence = 1.0
    return {"exact_error": float(f"{err_exact:.3e}"), "griffin_snr_db": r(snr, 2),
            "random_phase_snr_db": r(snr_rand, 2), "iters": 60,
            "spec_gap": r(spec_gap, 5), "spec_gap_random": r(spec_gap_rand, 5),
            "spec_gap_silence": spec_gap_silence,
            "win": win, "hop": hop, "n": int(len(y)),
            "audio_original": b64(y * 0.9), "audio_griffin": b64(gl * 0.9),
            "audio_random": b64(rand * 0.9),
            "marks": [{"sym": s, "start": int(a), "end": int(b)} for s, a, b in marks]}


# ---------------------------------------------------------------------------
# Stage 3: the filterbank, and the coefficients
# ---------------------------------------------------------------------------
def stage_mel():
    bank = mel_bank()
    freqs = np.fft.rfftfreq(NFFT, 1 / SR)
    # what it costs: rebuild the log spectrum from the log mel energies by
    # least squares, on real frames
    y, _ = utterance(list("aeiou"), seed=SEED + 1)
    frames, _, _ = stft(y, 256, 64, NFFT)
    mag = np.abs(frames)
    keep = mag.max(axis=1) > 1e-3
    mag = mag[keep]
    logspec = np.log(np.maximum(mag ** 2, 1e-10))
    logmel = np.log(np.maximum(mag ** 2 @ bank.T, 1e-10))
    A, *_ = np.linalg.lstsq(logmel, logspec, rcond=None)
    resid = logspec - logmel @ A
    rel = float(np.sqrt(np.mean(resid ** 2)) / np.sqrt(np.mean((logspec - logspec.mean()) ** 2)))
    return {"n_mels": N_MELS, "n_bins": int(NFFT // 2 + 1), "nfft": NFFT, "sr": SR,
            "bank": r(bank[:, :NFFT // 2 + 1], 5), "freqs": r(freqs, 2),
            "rebuild_rel_error": r(rel, 4), "n_frames": int(mag.shape[0]),
            "rank": int(np.linalg.matrix_rank(bank))}


def stage_cepstrum():
    bank = mel_bank()
    pitches = list(range(90, 231, 14))
    rows = []
    per_vowel = {}
    for sym in VOWELS:
        specs = []
        coefs = []
        for f0 in pitches:
            y = vowel(sym, np.full(int(0.25 * SR), float(f0)))
            frames, freqs, _ = stft(y, 256, 64, NFFT)
            mag = np.abs(frames).mean(axis=0)
            c, logmel = mfcc_of(mag, bank)
            specs.append(np.log(np.maximum(mag ** 2, 1e-10)))
            coefs.append(c)
        specs = np.array(specs)
        coefs = np.array(coefs)
        # spread across pitch, in units of the spread across vowels later
        per_vowel[sym] = {"spec": specs.mean(axis=0), "coef": coefs.mean(axis=0),
                          "spec_spread": float(np.mean(np.std(specs, axis=0))),
                          "coef_spread": float(np.mean(np.std(coefs[:, 1:], axis=0)))}
        rows.append({"vowel": sym,
                     "spec_spread": r(per_vowel[sym]["spec_spread"], 4),
                     "coef_spread": r(per_vowel[sym]["coef_spread"], 4)})

    # how far apart two different vowels are, in the same units
    syms = list(VOWELS)
    spec_between = np.mean([np.mean(np.abs(per_vowel[a]["spec"] - per_vowel[b]["spec"]))
                            for i, a in enumerate(syms) for b in syms[i + 1:]])
    coef_between = np.mean([np.mean(np.abs(per_vowel[a]["coef"][1:] - per_vowel[b]["coef"][1:]))
                            for i, a in enumerate(syms) for b in syms[i + 1:]])
    spec_within = float(np.mean([per_vowel[s]["spec_spread"] for s in syms]))
    coef_within = float(np.mean([per_vowel[s]["coef_spread"] for s in syms]))

    # reading the pitch back out of the high quefrencies, against the truth
    pitch_rows = []
    for f0 in pitches:
        y = vowel("a", np.full(int(0.25 * SR), float(f0)))
        frames, freqs, _ = stft(y, 512, 128, 1024)
        mag = np.abs(frames).mean(axis=0)
        logspec = np.log(np.maximum(mag, 1e-10))
        cep = np.fft.irfft(logspec)
        lo = int(SR / 300)
        hi = int(SR / 70)
        q = int(np.argmax(cep[lo:hi]) + lo)
        pitch_rows.append({"true": f0, "found": r(SR / q, 2), "quefrency": q})
    errs = np.array([abs(p["found"] - p["true"]) for p in pitch_rows])
    perr = float(errs.mean())
    # Two of the eleven come back an octave low, which is the classic failure of
    # this estimator and not a bug in it, so the mean is the wrong summary: the
    # median says what it does when it works and the count says how often it
    # does not.
    pmed = float(np.median(errs))
    octave = int(np.sum(errs > 10))

    demo = {}
    for sym in ["a", "i"]:
        for f0 in [100, 200]:
            y = vowel(sym, np.full(int(0.25 * SR), float(f0)))
            frames, freqs, _ = stft(y, 256, 64, NFFT)
            mag = np.abs(frames).mean(axis=0)
            c, logmel = mfcc_of(mag, bank)
            demo[f"{sym}{f0}"] = {
                "db": r(20 * np.log10(mag + 1e-8), 3),
                "logmel": r(logmel, 4), "coef": r(c, 4),
                "audio": b64(y / (np.max(np.abs(y)) + 1e-9) * 0.8),
            }
    return {"pitches": pitches, "rows": rows,
            "spec_between": r(spec_between, 4), "coef_between": r(coef_between, 4),
            "spec_within": r(spec_within, 4), "coef_within": r(coef_within, 4),
            "spec_ratio": r(spec_between / max(spec_within, 1e-9), 3),
            "coef_ratio": r(coef_between / max(coef_within, 1e-9), 3),
            "pitch": {"rows": pitch_rows, "mean_abs_error_hz": r(perr, 3),
                      "median_error_hz": r(pmed, 3), "octave_errors": octave,
                      "n": len(pitch_rows)},
            "demo": demo, "freqs": r(np.fft.rfftfreq(NFFT, 1 / SR), 2),
            "envelope": {s: r(envelope_db(np.fft.rfftfreq(NFFT, 1 / SR), VOWELS[s]), 3)
                         for s in ["a", "i"]}}


def stage_identify():
    """Can a vowel be identified from thirteen numbers, at a pitch never heard?

    In clean conditions every feature set gets all twenty five right, which
    says nothing, so the question is asked again with noise added at four
    levels and with the coefficients cut back one at a time. Both sweeps are
    the measurement; the clean row is only there to show it was too easy.
    """
    bank = mel_bank()
    train_p = [95, 115, 135, 155, 175]
    test_p = [105, 125, 145, 165, 205]
    syms = list(VOWELS)
    rng = np.random.default_rng(SEED)

    def features(sym, f0, snr_db=None, seed=0):
        y = vowel(sym, np.full(int(0.25 * SR), float(f0)))
        y = y / (np.std(y) + 1e-9)
        if snr_db is not None:
            g = np.random.default_rng(SEED + seed)
            y = y + g.normal(0, 10 ** (-snr_db / 20), len(y))
        frames, _, _ = stft(y, 256, 64, NFFT)
        mag = np.abs(frames).mean(axis=0)
        c, _ = mfcc_of(mag, bank)
        return c, np.log(np.maximum(mag ** 2, 1e-10))

    train = [(s, f0, *features(s, f0)) for s in syms for f0 in train_p]

    def score(take, snr_db, reps=4):
        right = total = 0
        for rep in range(reps):
            for s in syms:
                for f0 in test_p:
                    c, sp = features(s, f0, snr_db, seed=rep * 97 + f0)
                    q = take(c, sp)
                    best = min(train, key=lambda t: np.linalg.norm(take(t[2], t[3]) - q))
                    right += int(best[0] == s)
                    total += 1
        return right, total

    takes = [("thirteen MFCCs, the first dropped", lambda c, s: c[1:]),
             ("all thirteen MFCCs", lambda c, s: c),
             ("the whole log spectrum", lambda c, s: s)]
    rows = []
    for name, take in takes:
        entry = {"features": name}
        for label, snr in [("clean", None), ("25 dB", 25), ("20 dB", 20), ("15 dB", 15),
                           ("10 dB", 10)]:
            right, total = score(take, snr, reps=1 if snr is None else 4)
            entry[label] = r(right / total, 4)
            entry[f"{label}_n"] = total
        rows.append(entry)

    # And how many coefficients are actually needed. At 10 dB every feature set
    # is already at chance, so a sweep there measures nothing: it is run at the
    # level where the answer still varies.
    cut = []
    for k in range(2, N_MFCC + 1):
        right, total = score(lambda c, s: c[1:k], 20, reps=4)
        cut.append({"coefficients": k - 1, "accuracy": r(right / total, 4)})
    return {"rows": rows, "cut": cut, "cut_snr": 20, "train_pitches": train_p,
            "test_pitches": test_p, "chance": r(1 / len(syms), 4),
            "n_vowels": len(syms),
            "levels": ["clean", "25 dB", "20 dB", "15 dB", "10 dB"]}


def main():
    import scipy

    print("resolution")
    res = cached("resolution", stage_resolution)
    print("the sweep")
    ch = cached("chirp", stage_chirp)
    print("inversion")
    inv = cached("inversion", stage_inversion)
    print("the filterbank")
    mel = cached("mel", stage_mel)
    print("the cepstrum")
    cep = cached("cepstrum", stage_cepstrum)
    print("identification")
    ident = cached("identify", stage_identify)

    data = {
        "meta": {
            "sr": SR, "nfft": NFFT, "n_mels": N_MELS, "n_mfcc": N_MFCC, "seed": SEED,
            "vowels": {k: list(v) for k, v in VOWELS.items()},
            "fricatives": {k: list(v) for k, v in FRICATIVES.items()},
            "bandwidths": list(BANDWIDTHS),
            "numpy": np.__version__, "scipy": scipy.__version__,
            "threads": min(4, os.cpu_count() or 4),
        },
        "resolution": res, "chirp": ch, "inversion": inv, "mel": mel,
        "cepstrum": cep, "identify": ident,
    }
    (OUT / "spectrograms.json").write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {OUT / 'spectrograms.json'} "
          f"({(OUT / 'spectrograms.json').stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
