"""Everything article 4.3b cites, measured.

Two ways to put a neural network on sound, and the two things each is accused
of.

 1. A stack of dilated causal convolutions has a receptive field of exactly
    1 + sum of the dilations, which is a fact about the architecture and not
    about the data, so how deep the stack has to be before it can hear a pitch
    period is predictable before any training happens. Five depths are trained
    and the prediction is checked against the measurement.
 2. Sound is stored as sixteen bit numbers and generated as a choice among 256,
    so the compander in between is a real loss and is measured in decibels
    against the linear alternative.
 3. A model trained to predict the next sample from the true previous ones is
    not the model that generates: at generation time it is fed its own output.
    That gap is measured by letting the same network run free and comparing the
    spectrum of what it produces with the spectrum of what it was trained on.
 4. On the recognition side, the same log mel picture from article 4.3a goes to
    two decoders on the same utterances: a connectionist temporal classification
    head, which cannot invent symbols out of order, and an attention decoder,
    which can and does. Both are scored for symbol error rate, and the decoder
    is scored for the failure it is famous for, which is not stopping.
 5. And the boundaries: because the utterances were written down rather than
    annotated, an alignment can be scored in milliseconds against the truth.

Runs in about fifteen minutes on four cores. Stages cached under
~/.atlas_vision_data/wavenet_cache/.
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
OUT = ROOT / "wavenet" / "data"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path.home() / ".atlas_vision_data" / "wavenet_cache"
CACHE.mkdir(parents=True, exist_ok=True)

from src.utils.audio_data import (  # noqa: E402
    FRICATIVES, SR, VOWELS, to_int16, utterance, vowel,
)

SEED = 47
MU = 255
QUANT = 256
THREADS = min(4, os.cpu_count() or 4)
SYMBOLS = list(VOWELS) + list(FRICATIVES)
WIN, HOP, NFFT, N_MELS = 256, 80, 512, 26


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


def torch_setup():
    import torch
    torch.set_num_threads(THREADS)
    torch.manual_seed(SEED)
    return torch


# ---------------------------------------------------------------------------
# The compander, and what it costs
# ---------------------------------------------------------------------------
def mu_encode(x, mu=MU):
    x = np.clip(x, -1, 1)
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    return np.clip(((y + 1) / 2 * mu + 0.5).astype(int), 0, mu)


def mu_decode(q, mu=MU):
    y = 2 * (q.astype(float) / mu) - 1
    return np.sign(y) * (np.expm1(np.abs(y) * np.log1p(mu))) / mu


def snr_db(x, y):
    n = min(len(x), len(y))
    num = float(np.sum(x[:n] ** 2))
    den = float(np.sum((x[:n] - y[:n]) ** 2))
    return 10 * np.log10(num / max(den, 1e-30))


def stage_quantise():
    y, marks = utterance(list("aSieo"), seed=SEED)
    lin = np.round((y + 1) / 2 * 255).astype(int)
    lin_back = lin / 255 * 2 - 1
    mu_back = mu_decode(mu_encode(y))
    rows = [{"scheme": "linear, 8 bits", "snr_db": r(snr_db(y, lin_back), 2)},
            {"scheme": "mu-law, 8 bits", "snr_db": r(snr_db(y, mu_back), 2)}]
    # and where the difference lives: the quiet parts
    quiet = np.abs(y) < 0.1
    rows.append({"scheme": "linear, quiet samples only",
                 "snr_db": r(snr_db(y[quiet], lin_back[quiet]), 2)})
    rows.append({"scheme": "mu-law, quiet samples only",
                 "snr_db": r(snr_db(y[quiet], mu_back[quiet]), 2)})
    curve = np.linspace(-1, 1, 201)
    return {"rows": rows, "levels": QUANT, "mu": MU,
            "curve": {"x": r(curve, 4),
                      "mu": r(np.sign(curve) * np.log1p(MU * np.abs(curve)) / np.log1p(MU), 4)},
            "quiet_share": r(float(quiet.mean()), 4),
            "audio_original": b64(y * 0.9), "audio_linear": b64(lin_back * 0.9),
            "audio_mu": b64(mu_back * 0.9), "n": int(len(y))}


# ---------------------------------------------------------------------------
# The stack, and its receptive field
# ---------------------------------------------------------------------------
def make_wavenet(torch, layers, channels=32, quant=QUANT):
    nn = torch.nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(quant, channels)
            self.dilations = [2 ** i for i in range(layers)]
            self.filters = nn.ModuleList([
                nn.Conv1d(channels, 2 * channels, 2, dilation=d) for d in self.dilations])
            self.res = nn.ModuleList([nn.Conv1d(channels, channels, 1) for _ in self.dilations])
            self.skip = nn.ModuleList([nn.Conv1d(channels, channels, 1) for _ in self.dilations])
            self.head = nn.Sequential(nn.ReLU(), nn.Conv1d(channels, channels, 1),
                                      nn.ReLU(), nn.Conv1d(channels, quant, 1))
            self.receptive = 1 + sum(self.dilations)

        def forward(self, q):
            h = self.embed(q).transpose(1, 2)
            skips = 0
            for conv, res, skip, d in zip(self.filters, self.res, self.skip, self.dilations):
                pad = torch.nn.functional.pad(h, (d, 0))
                z = conv(pad)
                a, b = z.chunk(2, dim=1)
                z = torch.tanh(a) * torch.sigmoid(b)
                skips = skips + skip(z)
                h = h + res(z)
            return self.head(skips)

    return Net()


def training_audio(seconds=6.0, seed=SEED):
    """A steady vowel with a slowly drifting pitch: periodic enough that a
    receptive field shorter than one period is provably not enough."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    f0 = 120 + 8 * np.sin(2 * np.pi * np.arange(n) / (2.0 * SR))
    y = vowel("a", f0)
    y = y / (np.max(np.abs(y)) + 1e-9)
    return y, float(np.mean(f0))


def stage_receptive():
    torch = torch_setup()
    y, f0 = training_audio()
    q = mu_encode(y)
    period = SR / f0
    n_train = int(0.8 * len(q))
    seq = 1024
    rows = []
    for layers in [4, 5, 6, 7, 8]:
        torch.manual_seed(SEED)
        net = make_wavenet(torch, layers)
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        lossf = torch.nn.CrossEntropyLoss()
        g = np.random.default_rng(SEED)
        for step in range(220):
            starts = g.integers(0, n_train - seq - 1, size=8)
            batch = np.stack([q[s:s + seq + 1] for s in starts])
            xb = torch.tensor(batch[:, :-1], dtype=torch.long)
            yb = torch.tensor(batch[:, 1:], dtype=torch.long)
            opt.zero_grad()
            out = net(xb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
        # held out: the last fifth of the clip, teacher forced
        with torch.no_grad():
            xs = torch.tensor(q[n_train:n_train + 4096][None, :-1], dtype=torch.long)
            ys = torch.tensor(q[n_train + 1:n_train + 4096][None, :], dtype=torch.long)
            out = net(xs)
            nll = float(lossf(out, ys))
            pred = out.argmax(dim=1).numpy()[0]
            acc = float((pred == ys.numpy()[0]).mean())
            recon = mu_decode(pred)
            truth = mu_decode(ys.numpy()[0])
        rows.append({"layers": layers, "receptive": int(net.receptive),
                     "receptive_ms": r(1000 * net.receptive / SR, 2),
                     "nll": r(nll, 4), "accuracy": r(acc, 4),
                     "snr_db": r(snr_db(truth, recon), 2),
                     "params": int(sum(p.numel() for p in net.parameters())),
                     "covers_period": bool(net.receptive >= period)})
        print(f"    {layers} layers, receptive {net.receptive}: nll {nll:.4f}")
    return {"rows": rows, "f0": r(f0, 2), "period": r(period, 2),
            "period_ms": r(1000 * period / SR, 2), "seconds": 6.0,
            "audio": b64(y * 0.9), "n": int(len(y))}


def stage_free_running():
    """What the model does when it is fed its own output."""
    torch = torch_setup()
    y, f0 = training_audio()
    q = mu_encode(y)
    n_train = int(0.8 * len(q))
    net = make_wavenet(torch, 8)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    lossf = torch.nn.CrossEntropyLoss()
    g = np.random.default_rng(SEED)
    seq = 1024
    for step in range(600):
        starts = g.integers(0, n_train - seq - 1, size=8)
        batch = np.stack([q[s:s + seq + 1] for s in starts])
        xb = torch.tensor(batch[:, :-1], dtype=torch.long)
        yb = torch.tensor(batch[:, 1:], dtype=torch.long)
        opt.zero_grad()
        loss = lossf(net(xb), yb)
        loss.backward()
        opt.step()

    prime = q[n_train:n_train + 512]
    out = {}
    for temp in [0.0, 0.7, 1.0]:
        torch.manual_seed(SEED)
        hist = list(prime)
        with torch.no_grad():
            for _ in range(2000):
                ctx = torch.tensor(np.array(hist[-net.receptive - 1:])[None, :], dtype=torch.long)
                logits = net(ctx)[0, :, -1]
                if temp <= 0:
                    nxt = int(torch.argmax(logits))
                else:
                    p = torch.softmax(logits / temp, dim=0)
                    nxt = int(torch.multinomial(p, 1))
                hist.append(nxt)
        gen = mu_decode(np.array(hist[len(prime):]))
        ref = mu_decode(q[n_train + 512:n_train + 512 + 2000])
        # spectral distance rather than sample distance: two periodic signals
        # out of phase are far apart sample by sample and identical to the ear
        def spec(sig):
            w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(512) / 512)
            frames = [np.abs(np.fft.rfft(sig[s:s + 512] * w))
                      for s in range(0, len(sig) - 512, 256)]
            return np.log(np.mean(np.array(frames), axis=0) + 1e-8)
        d = float(np.mean(np.abs(spec(gen) - spec(ref))))
        zero_cross = float(np.mean(np.abs(np.diff(np.sign(gen))) > 0))
        out[str(temp)] = {"spec_distance": r(d, 4), "audio": b64(gen * 0.9),
                          "rms": r(float(np.sqrt(np.mean(gen ** 2))), 4),
                          "zero_crossing_rate": r(zero_cross, 4),
                          "wave": r(gen[:600], 4)}
    ref = mu_decode(q[n_train + 512:n_train + 512 + 2000])
    out["truth"] = {"audio": b64(ref * 0.9), "wave": r(ref[:600], 4),
                    "rms": r(float(np.sqrt(np.mean(ref ** 2))), 4),
                    "zero_crossing_rate": r(float(np.mean(np.abs(np.diff(np.sign(ref))) > 0)), 4)}
    return {"arms": out, "primed_with": 512, "generated": 2000,
            "receptive": int(net.receptive)}


# ---------------------------------------------------------------------------
# Recognition: the same picture, two decoders
# ---------------------------------------------------------------------------
def mel_bank(n_mels=N_MELS, nfft=NFFT, sr=SR):
    def to_mel(f):
        return 2595 * np.log10(1 + f / 700)

    def to_hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    pts = to_hz(np.linspace(to_mel(0), to_mel(sr / 2), n_mels + 2))
    bins = np.floor((nfft + 1) * pts / sr).astype(int)
    bank = np.zeros((n_mels, nfft // 2 + 1))
    for m in range(1, n_mels + 1):
        lo, mid, hi = bins[m - 1], bins[m], bins[m + 1]
        if mid > lo:
            bank[m - 1, lo:mid] = (np.arange(lo, mid) - lo) / (mid - lo)
        if hi > mid:
            bank[m - 1, mid:hi] = (hi - np.arange(mid, hi)) / (hi - mid)
    return bank


BANK = mel_bank()


def logmel(y):
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WIN) / WIN)
    frames = np.array([np.abs(np.fft.rfft(y[s:s + WIN] * w, NFFT)) ** 2
                       for s in range(0, len(y) - WIN + 1, HOP)])
    return np.log(np.maximum(frames @ BANK.T, 1e-10))


def make_corpus(n=500, seed=SEED, min_len=3, max_len=6):
    rng = np.random.default_rng(seed)
    items = []
    for i in range(n):
        k = int(rng.integers(min_len, max_len + 1))
        syms = [SYMBOLS[int(rng.integers(0, len(SYMBOLS)))] for _ in range(k)]
        y, marks = utterance(syms, seed=seed * 1000 + i)
        items.append({"y": y, "syms": syms, "marks": marks, "feat": logmel(y)})
    return items


def stage_recognition():
    torch = torch_setup()
    nn = torch.nn
    items = make_corpus()
    split = int(0.85 * len(items))
    train, test = items[:split], items[split:]
    n_sym = len(SYMBOLS)

    class Encoder(nn.Module):
        def __init__(self, hidden=96):
            super().__init__()
            self.rnn = nn.GRU(N_MELS, hidden, num_layers=2, batch_first=True,
                              bidirectional=True)
            self.out = nn.Linear(2 * hidden, hidden)

        def forward(self, x):
            h, _ = self.rnn(x)
            return torch.relu(self.out(h))

    class Ctc(nn.Module):
        def __init__(self, hidden=96):
            super().__init__()
            self.enc = Encoder(hidden)
            self.head = nn.Linear(hidden, n_sym + 1)      # + blank

        def forward(self, x):
            return self.head(self.enc(x))

    class AttnDecoder(nn.Module):
        """A miniature of the Whisper decoder: attends over the encoder output
        and emits symbols one at a time, including an end symbol it has to
        learn to produce."""

        def __init__(self, hidden=96):
            super().__init__()
            self.enc = Encoder(hidden)
            self.emb = nn.Embedding(n_sym + 2, hidden)    # + start, + end
            self.rnn = nn.GRU(2 * hidden, hidden, batch_first=True)
            self.out = nn.Linear(hidden, n_sym + 1)       # + end

        def step(self, tok, state, mem):
            e = self.emb(tok)
            q = e if state is None else state[-1]
            att = torch.softmax(torch.bmm(mem, q.unsqueeze(2)).squeeze(2), dim=1)
            ctx = torch.bmm(att.unsqueeze(1), mem)
            o, state = self.rnn(torch.cat([e.unsqueeze(1), ctx], dim=2), state)
            return self.out(o.squeeze(1)), state, att

    def pad(batch):
        m = max(b["feat"].shape[0] for b in batch)
        x = np.zeros((len(batch), m, N_MELS), dtype=np.float32)
        lens = []
        for i, b in enumerate(batch):
            f = b["feat"]
            x[i, :f.shape[0]] = f
            lens.append(f.shape[0])
        return torch.tensor(x), lens

    # --- CTC ---------------------------------------------------------------
    torch.manual_seed(SEED)
    ctc = Ctc()
    opt = torch.optim.Adam(ctc.parameters(), lr=2e-3)
    ctc_loss = nn.CTCLoss(blank=n_sym, zero_infinity=True)
    g = np.random.default_rng(SEED)
    for ep in range(14):
        order = g.permutation(len(train))
        for i in range(0, len(order), 8):
            batch = [train[j] for j in order[i:i + 8]]
            x, lens = pad(batch)
            logits = ctc(x).log_softmax(dim=2).transpose(0, 1)
            targets = torch.tensor([SYMBOLS.index(s) for b in batch for s in b["syms"]],
                                   dtype=torch.long)
            tlens = torch.tensor([len(b["syms"]) for b in batch])
            opt.zero_grad()
            loss = ctc_loss(logits, targets, torch.tensor(lens), tlens)
            loss.backward()
            opt.step()

    # --- attention decoder -------------------------------------------------
    torch.manual_seed(SEED)
    dec = AttnDecoder()
    opt2 = torch.optim.Adam(dec.parameters(), lr=2e-3)
    ce = nn.CrossEntropyLoss()
    END = n_sym
    START = n_sym + 1
    for ep in range(14):
        order = g.permutation(len(train))
        for i in range(0, len(order), 8):
            batch = [train[j] for j in order[i:i + 8]]
            x, lens = pad(batch)
            mem = dec.enc(x)
            k = max(len(b["syms"]) for b in batch)
            tgt = torch.full((len(batch), k + 1), END, dtype=torch.long)
            for bi, b in enumerate(batch):
                for si, s in enumerate(b["syms"]):
                    tgt[bi, si] = SYMBOLS.index(s)
            tok = torch.full((len(batch),), START, dtype=torch.long)
            state = None
            opt2.zero_grad()
            loss = 0
            for t in range(k + 1):
                logits, state, _ = dec.step(tok, state, mem)
                loss = loss + ce(logits, tgt[:, t])
                tok = tgt[:, t].clamp(max=n_sym)          # teacher forcing
            (loss / (k + 1)).backward()
            opt2.step()

    # --- scoring -----------------------------------------------------------
    def edit(a, b):
        d = np.zeros((len(a) + 1, len(b) + 1), dtype=int)
        d[:, 0] = np.arange(len(a) + 1)
        d[0, :] = np.arange(len(b) + 1)
        for i in range(1, len(a) + 1):
            for j in range(1, len(b) + 1):
                d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                              d[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
        return int(d[-1, -1])

    ctc_err = ctc_len = 0
    # Two references for the same spikes, because picking one of them silently
    # decides the answer. The first version of this measured against the middle
    # of each phone and reported 57,5 ms against a frame step of 10, which reads
    # as an alignment five frames wrong. It is not: the spikes land 3 to 16 ms
    # after the phone STARTS. A connectionist temporal classification loss has
    # no term that would put a spike in the middle of anything, so the middle is
    # the wrong reference, and the article now shows both and says which.
    onset_err = []
    centre_err = []
    signed_centre = []
    durations = []
    with torch.no_grad():
        for b in test:
            x, _ = pad([b])
            probs = ctc(x).softmax(dim=2)[0].numpy()
            best = probs.argmax(axis=1)
            out = []
            peaks = []
            prev = -1
            for t, k in enumerate(best):
                if k != prev and k != n_sym:
                    out.append(SYMBOLS[k])
                    peaks.append(t)
                prev = k
            ctc_err += edit(out, b["syms"])
            ctc_len += len(b["syms"])
            if len(out) == len(b["syms"]):
                for t, (sym, start, end) in zip(peaks, b["marks"]):
                    spike = (t * HOP + WIN / 2) / SR
                    centre = (start + end) / 2 / SR
                    onset_err.append(abs(spike - start / SR) * 1000)
                    centre_err.append(abs(spike - centre) * 1000)
                    signed_centre.append((spike - centre) * 1000)
                    durations.append((end - start) / SR * 1000)

    dec_err = dec_len = 0
    too_long = 0
    repeats = 0
    lengths = []
    with torch.no_grad():
        for b in test:
            x, _ = pad([b])
            mem = dec.enc(x)
            tok = torch.tensor([START])
            state = None
            out = []
            for _ in range(24):                    # a hard cap, well past any truth
                logits, state, _ = dec.step(tok, state, mem)
                k = int(logits.argmax())
                if k == END:
                    break
                out.append(SYMBOLS[k])
                tok = torch.tensor([k])
            dec_err += edit(out, b["syms"])
            dec_len += len(b["syms"])
            lengths.append(len(out))
            if len(out) > len(b["syms"]):
                too_long += 1
            for i in range(len(out) - 2):
                if out[i] == out[i + 1] == out[i + 2]:
                    repeats += 1
                    break

    demo = test[0]
    with torch.no_grad():
        x, _ = pad([demo])
        probs = ctc(x).softmax(dim=2)[0].numpy()
    return {
        "n_train": len(train), "n_test": len(test), "symbols": SYMBOLS,
        "ctc": {"ser": r(ctc_err / ctc_len, 4), "errors": ctc_err, "symbols": ctc_len},
        "decoder": {"ser": r(dec_err / dec_len, 4), "errors": dec_err, "symbols": dec_len,
                    "too_long": too_long, "runaway": repeats,
                    "mean_length": r(float(np.mean(lengths)), 3),
                    "cap": 24},
        "alignment": {"n": len(onset_err),
                      "median_ms": r(float(np.median(onset_err)) if onset_err else 0, 2),
                      "mean_ms": r(float(np.mean(onset_err)) if onset_err else 0, 2),
                      "centre_median_ms": r(float(np.median(centre_err)) if centre_err else 0, 2),
                      "signed_centre_ms": r(float(np.median(signed_centre))
                                            if signed_centre else 0, 2),
                      "phone_median_ms": r(float(np.median(durations)) if durations else 0, 2),
                      "frame_ms": r(1000 * HOP / SR, 2)},
        "demo": {"syms": demo["syms"],
                 "marks": [{"sym": s, "start": int(a), "end": int(bb)}
                           for s, a, bb in demo["marks"]],
                 "posteriors": r(probs, 4),
                 "feat": r(demo["feat"], 3),
                 "audio": b64(demo["y"] * 0.9),
                 "frame_ms": r(1000 * HOP / SR, 2)},
        "frames_per_second": r(SR / HOP, 1),
    }


def main():
    import torch

    print("the compander")
    quant = cached("quantise", stage_quantise)
    print("the receptive field")
    rec = cached("receptive", stage_receptive)
    print("free running")
    free = cached("free_running", stage_free_running)
    print("recognition")
    asr = cached("recognition", stage_recognition)

    data = {
        "meta": {
            "sr": SR, "seed": SEED, "mu": MU, "levels": QUANT,
            "win": WIN, "hop": HOP, "nfft": NFFT, "n_mels": N_MELS,
            "symbols": SYMBOLS, "vowels": {k: list(v) for k, v in VOWELS.items()},
            "threads": THREADS, "torch": torch.__version__, "numpy": np.__version__,
        },
        "quantise": quant, "receptive": rec, "free": free, "recognition": asr,
    }
    (OUT / "wavenet.json").write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {OUT / 'wavenet.json'} "
          f"({(OUT / 'wavenet.json').stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
