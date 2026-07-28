"""The three thumbnails for the module 2.3 and 2.4 cards.

Each one is the picture its own article draws, reduced to the size of a card,
and each one is built from that article's shipped JSON rather than from a fresh
computation, so a card cannot drift away from the page it advertises.

A card is about 460 px wide and gets looked at for a second, so the rule from
the previous batch applies: thin until the marks separate, group the circles by
fill so the colour is written once, and round the coordinates to integers. The
other thirty-two thumbnails live in the 2-9 kB band and these have to as well,
because the front page loads all of them at once.

And the rule that came out of the module 2.2 batch: **a thumbnail that
illustrates a measured property is asserted, not just drawn.** The lattice below
claims that most of the sets are not frequent, and the ellipse below claims that
the rectangle's corner is outside it; both are checked before the file is
written.

Run from anywhere: `python src/utils/generate_web_thumbnails_2_3.py`.
"""
import base64
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets" / "thumbnails"
W, H = 400, 300
BG = "#f1f3f3"
INK = "#232f3e"


def head(comment):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
            f'  <rect width="{W}" height="{H}" fill="{BG}" />'] + [f'  <!-- {c} -->' for c in comment]


def write(name, lines):
    path = OUT / f"{name}.svg"
    path.write_text("\n".join(lines) + "\n</svg>\n", encoding="utf-8")
    print(f"  {path.name}: {path.stat().st_size / 1024:.1f} kB")


def group(points, fill, r, opacity=None, stroke=None, sw=0.6):
    """Circles sharing one fill, written as one group so the colour appears once."""
    op = "" if opacity is None else f' opacity="{opacity}"'
    st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    out = [f'  <g fill="{fill}"{st}{op}>']
    row = []
    for x, y in points:
        row.append(f'<circle cx="{round(x)}" cy="{round(y)}" r="{r}"/>')
        if len(row) == 6:
            out.append("    " + "".join(row))
            row = []
    if row:
        out.append("    " + "".join(row))
    out.append("  </g>")
    return out


# =========================================================== 2.3a: odd one out
# The article's signature picture: two columns that correlate, the rectangle the
# two margins are willing to accept, the ellipse the pair is willing to accept,
# and the corner of the rectangle that is ordinary on each axis and impossible
# on the pair.
D = json.loads((ROOT / "outliers" / "data" / "outliers.json").read_text(encoding="utf-8"))
T = D["two_d"]
A = T["points"]
PRIMARY = "#1d4ed8"

xs = [p[0] for p in A]
ys = [p[1] for p in A]
corner = T["corner"]["point"]
lo = [min(min(xs), corner[0]) - 0.35, min(min(ys), corner[1]) - 0.25]
hi = [max(max(xs), corner[0]) + 0.35, max(max(ys), corner[1]) + 0.25]
sx = lambda v: 24 + (v - lo[0]) / (hi[0] - lo[0]) * (W - 48)
sy = lambda v: H - 22 - (v - lo[1]) / (hi[1] - lo[1]) * (H - 44)

mu, sd, zc = T["mu"], T["sd"], T["z_cut"]
rect = (sx(mu[0] - zc * sd[0]), sy(mu[1] + zc * sd[1]),
        sx(mu[0] + zc * sd[0]) - sx(mu[0] - zc * sd[0]),
        sy(mu[1] - zc * sd[1]) - sy(mu[1] + zc * sd[1]))

# the ellipse, from the covariance the article publishes
S = T["cov"]
tr, det = S[0][0] + S[1][1], S[0][0] * S[1][1] - S[0][1] ** 2
disc = math.sqrt(max(0.0, tr * tr / 4 - det))
l1, l2 = tr / 2 + disc, tr / 2 - disc
vx, vy = (1.0, 0.0) if abs(S[0][1]) < 1e-12 else (l1 - S[1][1], S[0][1])
nrm = math.hypot(vx, vy)
vx, vy = vx / nrm, vy / nrm
r1 = math.sqrt(max(0.0, T["chi_cut"] * l1))
r2 = math.sqrt(max(0.0, T["chi_cut"] * l2))
ring = []
for i in range(65):
    t = 2 * math.pi * i / 64
    u, v = r1 * math.cos(t), r2 * math.sin(t)
    ring.append((sx(mu[0] + u * vx - v * vy), sy(mu[1] + u * vy + v * vx)))

# the claim the picture makes, checked before it is drawn
d2_corner = T["corner"]["d"] ** 2
assert d2_corner > T["chi_cut"], "the corner of the rectangle is inside the ellipse after all"
assert T["rect_only"] > 0, "no bottle sits in the blind spot, so there is nothing to draw"

inside = [i for i in range(len(A)) if i not in T["rect_only_idx"]]
lines = head(["two columns that correlate at 0.86: the rectangle two margins accept,",
              "the ellipse the pair accepts, and the corner that is ordinary on each",
              "axis and impossible on both"])
lines.append(f'  <rect x="{round(rect[0])}" y="{round(rect[1])}" width="{round(rect[2])}" '
             f'height="{round(rect[3])}" fill="#df2a5d" fill-opacity="0.07" '
             f'stroke="#df2a5d" stroke-opacity="0.55" stroke-width="1.4"/>')
lines.append('  <path fill="' + PRIMARY + '" fill-opacity="0.1" stroke="' + PRIMARY
             + '" stroke-width="1.8" d="M'
             + "L".join(f"{round(x)} {round(y)}" for x, y in ring) + 'Z"/>')
lines += group([(sx(A[i][0]), sy(A[i][1])) for i in inside], INK, 2.6, opacity=0.45, stroke=BG)
lines += group([(sx(A[i][0]), sy(A[i][1])) for i in T["rect_only_idx"]], PRIMARY, 4.4, stroke=BG,
               sw=0.9)
cx, cy = sx(corner[0]), sy(corner[1])
lines.append(f'  <circle cx="{round(cx)}" cy="{round(cy)}" r="6.5" fill="{PRIMARY}"/>')
lines.append(f'  <circle cx="{round(cx)}" cy="{round(cy)}" r="11" fill="none" '
             f'stroke="{PRIMARY}" stroke-width="1.6" opacity="0.7"/>')
write("outliers", lines)

# ============================================================== 2.3b: alone
# Two densities, the twelve planted anomalies ringed, and a handful of the
# forest's own axis parallel cuts, which is the picture of what the detector
# has to work with and of the bias the article measures.
I = json.loads((ROOT / "isolation" / "data" / "isolation.json").read_text(encoding="utf-8"))
S2 = I["sets"]["two densities"]
P2, L2 = S2["points"], S2["labels"]
GREEN = "#065f46"
xs = [p[0] for p in P2]
ys = [p[1] for p in P2]
lo = [min(xs) - 0.5, min(ys) - 0.5]
hi = [max(xs) + 0.5, max(ys) + 0.5]
sx = lambda v: 20 + (v - lo[0]) / (hi[0] - lo[0]) * (W - 40)
sy = lambda v: H - 18 - (v - lo[1]) / (hi[1] - lo[1]) * (H - 36)

# The cuts are the first few a real tree made, rebuilt with the same generator
# the page uses; a decorative grid would be a picture of an idea instead.
class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def unit(self):
        self.s = (self.s * 1664525 + 1013904223) & 0xFFFFFFFF
        return self.s / 4294967296.0

    def below(self, n):
        return int(self.unit() * n)


# The FIRST cut of seven independent trees, not the first seven cuts of one:
# following one tree down narrows the region it is cutting, so the last four
# land within a hair of each other and the card reads as a mistake instead of
# as a forest. Seven roots over the whole data spread out the way the picture
# needs and are just as real.
rng = LCG(11)
cuts = []
for _ in range(7):
    f = rng.below(2)
    vals = [p[f] for p in P2]
    a, b = min(vals), max(vals)
    cuts.append((f, a + rng.unit() * (b - a)))

lines = head(["a tight cluster, a loose one, twelve planted anomalies ringed, and the",
              "first cut of seven real isolation trees: every one of them axis parallel"])
for f, s in cuts:
    if f == 0:
        lines.append(f'  <line x1="{round(sx(s))}" y1="14" x2="{round(sx(s))}" y2="{H - 14}" '
                     f'stroke="{GREEN}" stroke-width="1.1" opacity="0.4"/>')
    else:
        lines.append(f'  <line x1="14" y1="{round(sy(s))}" x2="{W - 14}" y2="{round(sy(s))}" '
                     f'stroke="{GREEN}" stroke-width="1.1" opacity="0.4"/>')
# Thinned to every other ordinary point: the file has to stay in the same 2-9 kB
# band as the other thirty-two, and halving a cluster uniformly does not change
# which of the two is tight and which is loose, which is all the card claims.
ordinary = [i for i in range(len(P2)) if not L2[i] and i % 2 == 0]
lines += group([(sx(P2[i][0]), sy(P2[i][1])) for i in ordinary], INK, 3.0, opacity=0.6, stroke=BG)
for i in range(len(P2)):
    if L2[i]:
        lines.append(f'  <circle cx="{round(sx(P2[i][0]))}" cy="{round(sy(P2[i][1]))}" r="6" '
                     f'fill="none" stroke="#df2a5d" stroke-width="2"/>')
assert sum(L2) == 12, "the caption says twelve anomalies"
write("isolation", lines)

# =========================================================== 2.4: bought together
# The lattice of every subset of eight items, by size, with the frequent ones
# filled. The shape is the argument: fat in the middle, and almost none of it
# frequent.
B = json.loads((ROOT / "baskets" / "data" / "baskets.json").read_text(encoding="utf-8"))
G = B["groceries"]
PURPLE = "#701a75"
K = 8
NB = G["bytes"]


def bits(b64):
    raw = base64.b64decode(b64)
    return int.from_bytes(raw, "little")


masks = [bits(G["bitsets"][i]) for i in range(K)]
min_count = math.ceil(0.02 * G["n"])
rows = {s: [] for s in range(1, K + 1)}


def walk(start, mask, chosen):
    for j in range(start, K):
        m = mask & masks[j]
        rows[len(chosen) + 1].append(m.bit_count())
        walk(j + 1, m, chosen + [j])


walk(0, (1 << G["n"]) - 1, [])
total = sum(len(v) for v in rows.values())
freq = sum(1 for v in rows.values() for c in v if c >= min_count)
assert total == 2 ** K - 1, f"{total} subsets, expected {2 ** K - 1}"
assert 0 < freq < total / 3, "the picture claims most of the lattice is not frequent"

lines = head([f"every one of the {total} subsets of eight items, laid out by size:",
              f"only {freq} of them reach the threshold, and the lattice is fattest",
              "exactly where the counting would go"])
fill_pts, pale_pts = [], []
for size, counts in rows.items():
    y = 26 + (H - 52) * (size - 1) / (K - 1)
    for i, c in enumerate(counts):
        x = 18 + (W - 36) * (i + 0.5) / len(counts)
        (fill_pts if c >= min_count else pale_pts).append((x, y))
lines += group(pale_pts, "#c2cbcb", 2.4, opacity=0.9)
lines += group(fill_pts, PURPLE, 3.4, stroke=BG, sw=0.7)
write("baskets", lines)
print("\nthe three cards are drawn from the articles' own JSON, so they cannot drift")
