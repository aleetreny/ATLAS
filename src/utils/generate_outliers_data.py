"""Data for the ATLAS 'Odd One Out' web article, the first of module 2.3.

Three rules for calling a point strange, all of them older than computers, all
of them still the first thing anybody reaches for: how many standard deviations
from the mean, how far past Tukey's fences, and the multivariate version of the
first one, Mahalanobis distance. They share a defect that is easy to state and
almost never drawn: **the yardstick is estimated from the data that contains
the suspect**, so the suspect gets a vote at its own trial.

Everything here is measured on the 178 bottles of wine that articles 1, 12, 25
and 27 already publish, at three widths: one column (magnesium), two columns
(flavanoids against total phenols, which correlate at 0.86) and all thirteen.
The matrix is exported to four decimals, which is exact for every value in the
file bar one, so the browser holds the same numbers this script measured and
the guards can be tight.

Results worth knowing before reading the code:

  * a z-score has a hard ceiling that has nothing to do with the data. With n
    points, no z can exceed (n-1)/sqrt(n), so the three-sigma rule is
    arithmetically incapable of firing for n < 11. With k of the n points
    contaminated together the ceiling drops to sqrt((n-1)(n-k)/(nk)), and on
    these 178 wines eighteen contaminated bottles put every possible z below
    three, however far out they are.
  * the same thing happens in p dimensions: the squared Mahalanobis distance to
    the sample mean cannot exceed (n-1)^2/n, and the n distances sum to exactly
    p(n-1) no matter what the data is. That identity is an exact browser guard.
  * the chi-squared cutoff everybody uses for Mahalanobis is the cutoff for
    KNOWN mean and covariance. Estimated from the same sample it is a scaled
    beta, and the difference is measured here rather than asserted.
  * two columns that correlate disagree with their own margins. Four of the
    178 wines sit inside both marginal intervals and outside the joint ellipse;
    one does the opposite.

No wall clock is published anywhere in this article: nothing here is timed, and
the cost section counts operations, which are the same on every machine.

Run from anywhere: `python src/utils/generate_outliers_data.py`.
"""
import json
import math
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy import stats
from sklearn.datasets import load_wine

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outliers" / "data"
OUT.mkdir(parents=True, exist_ok=True)

DP = 4                        # decimals the browser will hold, and we measure on
rnd = lambda v, n=4: round(float(v), n)
SEED = 19                     # the module's seed, kept for continuity

print("=" * 70)
print("Odd One Out: the estimator is the suspect")
print("=" * 70)

# ============================================================ the wine, rounded
# Rule from the recipe: measure on the numbers the page will hold, not on the
# ones that came out of the loader. Here the two are the same to a part in a
# million, because the UCI file is written to two decimals; exactly one value
# (row 171, colour intensity 9.899999) needs the rounding at all.
wine = load_wine()
FEATURES = list(wine.feature_names)
X = np.round(wine.data, DP)
Y = wine.target
N, P = X.shape
raw_gap = float(np.abs(X - wine.data).max())
print(f"\n178 bottles, 13 columns, exported at {DP} decimals; the rounding moves "
      f"the worst value by {raw_gap:.1e}")

# The same bottles the rest of the site publishes. `distances/` ships the
# standardised matrix, so standardising ours has to reproduce it: reuse is
# demonstrated, not declared.
shared = {}
dist_path = ROOT / "distances" / "data" / "distances.json"
if dist_path.exists():
    ref = np.array(json.loads(dist_path.read_text(encoding="utf-8"))["wine"]["matrix"])
    ours = (X - X.mean(0)) / X.std(0)
    shared["distances_matrix"] = float(np.abs(ours - ref).max())
    print(f"  standardised, it reproduces the matrix `distances/` ships to "
          f"{shared['distances_matrix']:.2e}")
dirs_path = ROOT / "directions" / "data" / "directions.json"
if dirs_path.exists():
    cloud = json.loads(dirs_path.read_text(encoding="utf-8"))["cloud"]
    cols = [FEATURES.index(c) for c in cloud["features"]]
    pair = X[:, cols]
    ref = np.array(cloud["raw"]["points"])
    shared["directions_pair"] = {
        "gap": float(np.abs((pair - pair.mean(0)) - ref).max()),
        "cols": list(cloud["features"]),
    }
    print(f"  and, centred, the pair `directions/` draws "
          f"({', '.join(shared['directions_pair']['cols'])}) to "
          f"{shared['directions_pair']['gap']:.2e}")

# ================================================================ one column
# Magnesium: the column article 27 measured as the wine's most private one
# (83.72% of its variance is nobody else's business), which is exactly the kind
# of column somebody screens one at a time.
COL = "magnesium"
mg = X[:, FEATURES.index(COL)]


def classical(v):
    """Mean, sample sd, and the z of every point. Sample sd (n-1) because that
    is what every textbook rule and every library means by `sd`."""
    m, s = float(v.mean()), float(v.std(ddof=1))
    return m, s, (v - m) / s


def robust(v):
    """Median, MAD, and Iglewicz and Hoaglin's modified z. The 0.6745 makes MAD
    estimate the same thing as sd on gaussian data, so the two scores are on
    one scale and the comparison below means something."""
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    return med, mad, 0.6745 * (v - med) / mad


def tukey(v, whisker=1.5):
    """Quartiles by linear interpolation, which is numpy's default and what the
    browser has to reproduce exactly or the fences land between points."""
    q1, q3 = (float(q) for q in np.percentile(v, [25, 75]))
    iqr = q3 - q1
    return q1, q3, iqr, q1 - whisker * iqr, q3 + whisker * iqr


def tukey_gaussian_tail(whisker=1.5):
    """What fraction of a gaussian falls outside Tukey's fences.

    Quoted as "about 0.7%" everywhere, which is the kind of number this repo is
    not allowed to cite: the quartiles of a standard normal are at +-0.6745, the
    interquartile range is twice that, and the tail follows. Computed rather
    than remembered."""
    q = float(stats.norm.ppf(0.75))
    fence = q + whisker * 2 * q
    return float(2 * stats.norm.sf(fence)), fence


m_mean, m_sd, m_z = classical(mg)
m_med, m_mad, m_mz = robust(mg)
q1, q3, iqr, lo_f, hi_f = tukey(mg)
Z_CUT, MZ_CUT = 3.0, 3.5
one_d = {
    "column": COL,
    "values": [rnd(v) for v in mg],
    "mean": rnd(m_mean), "sd": rnd(m_sd),
    "median": rnd(m_med), "mad": rnd(m_mad),
    "q1": rnd(q1), "q3": rnd(q3), "iqr": rnd(iqr),
    "fence_lo": rnd(lo_f), "fence_hi": rnd(hi_f),
    "z_cut": Z_CUT, "mz_cut": MZ_CUT,
    "flag_z": [int(i) for i in np.where(np.abs(m_z) > Z_CUT)[0]],
    "flag_mz": [int(i) for i in np.where(np.abs(m_mz) > MZ_CUT)[0]],
    "flag_iqr": [int(i) for i in np.where((mg < lo_f) | (mg > hi_f))[0]],
    "max_z": rnd(float(np.abs(m_z).max())),
    "max_mz": rnd(float(np.abs(m_mz).max())),
    "sorted_top": [rnd(v) for v in np.sort(mg)[-8:]],
    "tukey_tail": rnd(tukey_gaussian_tail()[0], 5),
    "tukey_fence_sigma": rnd(tukey_gaussian_tail()[1], 4),
}
print(f"  outside the fences, a gaussian puts {100 * one_d['tukey_tail']:.3f}% of itself "
      f"(the fences sit at {one_d['tukey_fence_sigma']:.4f} sigma)")
print(f"\n{COL}: mean {m_mean:.4f}, sd {m_sd:.4f}, median {m_med:.0f}, MAD {m_mad:.0f}")
print(f"  fences {lo_f:.2f} and {hi_f:.2f}")
print(f"  flagged: z>{Z_CUT} {len(one_d['flag_z'])}, modified z>{MZ_CUT} "
      f"{len(one_d['flag_mz'])}, outside the fences {len(one_d['flag_iqr'])}")

# ---- the leave-one-out gap: how much the suspect flatters itself
loo = []
order = np.argsort(mg)[::-1]
for rank in range(4):
    i = int(order[rank])
    keep = np.ones(N, bool)
    keep[i] = False
    m2, s2 = mg[keep].mean(), mg[keep].std(ddof=1)
    loo.append({"index": i, "value": rnd(mg[i]), "z_in": rnd(m_z[i]),
                "z_out": rnd((mg[i] - m2) / s2)})
print("  each of the top four scored with and without itself in the yardstick:")
for r in loo:
    print(f"    {r['value']:>6.0f}: z {r['z_in']:.3f} in-sample, {r['z_out']:.3f} left out")

# ================================================================== the ceiling
# The whole article in one line of algebra. Put k of the n values at distance d
# and the rest at 0: the mean is kd/n, the sample sd is d*sqrt(k(n-k)/(n(n-1))),
# and the z of one of the k is sqrt((n-1)(n-k)/(nk)) with the d cancelled out.
# It does not depend on d at all, so no amount of pushing gets past it.
def ceiling(n, k=1):
    return math.sqrt((n - 1) * (n - k) / (n * k))


def measured_ceiling(n, k, d=1e9):
    v = np.zeros(n)
    v[:k] = d
    m, s = v.mean(), v.std(ddof=1)
    return float((v[0] - m) / s)


ceil_rows = []
for n in [5, 8, 10, 11, 12, 20, 30, 50, 100, N]:
    ceil_rows.append({"n": n, "k1": rnd(ceiling(n, 1), 4),
                      "measured": rnd(measured_ceiling(n, 1), 4)})
ceil_err = max(abs(r["k1"] - r["measured"]) for r in ceil_rows)
# scanned rather than read off the printed rows, so the sentence is about every
# n and not about the ten that happen to be in the table
first_n = next(n for n in range(2, 10000) if ceiling(n, 1) >= Z_CUT)
print(f"\nthe ceiling on a single z, (n-1)/sqrt(n):")
for r in ceil_rows:
    print(f"  n = {r['n']:>3}  ceiling {r['k1']:.4f}   pushed to 1e9: {r['measured']:.4f}")
print(f"  the formula and the push agree to {ceil_err:.1e}")
print(f"  the three-sigma rule cannot fire at all below n = {first_n}")

# with k contaminated points the ceiling drops. How many kill the rule outright?
k_rows = []
for k in range(1, 41):
    k_rows.append({"k": k, "ceiling": rnd(ceiling(N, k), 4),
                   "measured": rnd(measured_ceiling(N, k), 4)})
k_kills = min(r["k"] for r in k_rows if r["ceiling"] < Z_CUT)
# and the closed form for that k, from (n-1)(n-k) = 9nk
k_closed = N * (N - 1) / (10 * N - 1)
print(f"\nwith k of the {N} contaminated together the ceiling is sqrt((n-1)(n-k)/(nk)):")
for k in (1, 2, 5, 9, 18, 20, 40):
    print(f"  k = {k:>2}: {ceiling(N, k):.4f}")
print(f"  below three from k = {k_kills} on (closed form says {k_closed:.2f})")

# ---- what the widget draws: z of a suspect as it is dragged out, with 1 and 2
# of them moving. The second curve rises and then comes back down, which is
# masking with no words attached.
SUB_N = 12
sub_idx = list(range(SUB_N))                       # the first twelve bottles
sub = mg[sub_idx].copy()
SUS = SUB_N - 1                                    # the one the reader drags
COMPANION = SUB_N - 2                              # the one that sits still
base_span = float(sub.max() - sub.min())
# The drag starts where the suspect actually is, so the first frame of the
# widget is the sample as it comes, and runs out to twelve times the sample's
# own width. Twelve, not four: the point is that pushing further stops helping.
drag_grid = [rnd(float(v)) for v in
             np.linspace(float(sub[SUS]), float(sub.max()) + 12 * base_span, 81)]
# Where the companion is parked when there are two of them. Three widths out is
# enough to be flagged on its own (z just past three, which is nearly all the
# room a twelve-point sample has above three), and that is what makes it worth
# watching it stop being flagged.
COMP_AT = rnd(float(sub.max()) + 3 * base_span)


def drag_curve(values, positions, mode):
    """mode: 'one' drags a single point, 'two' drags one while a second sits at
    COMP_AT. Every score in the row is computed from the full sample as it
    stands at that position, which is the whole point of the picture."""
    out = []
    for pos in positions:
        v = values.copy()
        v[SUS] = pos
        if mode == "two":
            v[COMPANION] = COMP_AT
        m, s = v.mean(), v.std(ddof=1)
        med = np.median(v)
        mad = np.median(np.abs(v - med))
        qq1, qq3 = np.percentile(v, [25, 75])
        f_hi = qq3 + 1.5 * (qq3 - qq1)
        z = (v - m) / s
        mz = 0.6745 * (v - med) / mad if mad > 0 else np.full_like(v, np.inf)
        out.append({"pos": rnd(pos), "mean": rnd(m), "sd": rnd(s),
                    "z": rnd(z[SUS]), "mz": rnd(mz[SUS]),
                    "z_comp": rnd(z[COMPANION]) if mode == "two" else None,
                    "mz_comp": rnd(mz[COMPANION]) if mode == "two" else None,
                    "fence": rnd(f_hi), "past_fence": bool(pos > f_hi),
                    # what each rule would actually report, which is the number
                    # the reader is being asked to compare
                    "n_z": int((np.abs(z) > Z_CUT).sum()),
                    "n_mz": int((np.abs(mz) > MZ_CUT).sum()),
                    "n_fence": int(((v < qq1 - 1.5 * (qq3 - qq1)) | (v > f_hi)).sum())})
    return out


one_mover = drag_curve(sub, drag_grid, "one")
with_comp = drag_curve(sub, drag_grid, "two")
comp_peak = max(with_comp, key=lambda r: r["z_comp"])
comp_end = with_comp[-1]
print(f"\nthe {SUB_N}-bottle sample the widget opens on: {[int(v) for v in sub]}")
print(f"  one point dragged out: z saturates at {one_mover[-1]['z']:.4f} "
      f"(ceiling {ceiling(SUB_N, 1):.4f})")
print(f"  with a companion parked at {COMP_AT:.0f}, the companion's own z peaks at "
      f"{comp_peak['z_comp']:.4f} (with the suspect at {comp_peak['pos']:.0f}) and is down to "
      f"{comp_end['z_comp']:.4f} by the end of the drag")
print(f"    its modified z over the same drag: {with_comp[0]['mz_comp']:.4f} to "
      f"{comp_end['mz_comp']:.4f}")
# the crossing: where the companion stops being flagged by the classical rule
comp_lost = next((r["pos"] for r in with_comp if abs(r["z_comp"]) < Z_CUT), None)
crossed = [r for r in with_comp if abs(r["z_comp"]) >= Z_CUT]
print(f"    the companion is past three sigma in {len(crossed)} of the "
      f"{len(with_comp)} drag positions, and stops being flagged once the suspect passes "
      f"{comp_lost:.0f}" if comp_lost else "")
print(f"    counts at the two ends: z flags {with_comp[0]['n_z']} then "
      f"{with_comp[-1]['n_z']}; modified z flags {with_comp[0]['n_mz']} then "
      f"{with_comp[-1]['n_mz']}; the fences flag {with_comp[0]['n_fence']} then "
      f"{with_comp[-1]['n_fence']}")

one_d.update({
    "loo": loo,
    "ceiling": {"rows": ceil_rows, "err": ceil_err, "first_n": first_n,
                "k_rows": k_rows, "k_kills": k_kills, "k_closed": rnd(k_closed, 2)},
    "sub": {"n": SUB_N, "index": sub_idx, "values": [rnd(v) for v in sub],
            "suspect": SUS, "companion": COMPANION, "comp_at": COMP_AT,
            "grid": drag_grid, "one": one_mover, "two": with_comp,
            "ceil_one": rnd(ceiling(SUB_N, 1)), "ceil_two": rnd(ceiling(SUB_N, 2)),
            "comp_peak": comp_peak, "comp_end": comp_end,
            "comp_flagged_positions": len(crossed), "comp_lost_at": comp_lost},
})

# ============================================================ breakdown, measured
# Not the textbook fraction: the smallest number of points you have to move to
# send the estimate wherever you like, found by moving them and watching.
def breakdown(name, fn, values, target=1e6):
    v0 = fn(values)
    for k in range(1, len(values) + 1):
        v = values.copy()
        v[:k] = 1e12
        if abs(fn(v) - v0) > target:
            return {"name": name, "points": k, "fraction": rnd(k / len(values), 4),
                    "base": rnd(v0), "moved": rnd(fn(v))}
    return {"name": name, "points": None, "fraction": None, "base": rnd(v0), "moved": None}


bd = [
    breakdown("mean", lambda v: float(v.mean()), mg.copy()),
    breakdown("standard deviation", lambda v: float(v.std(ddof=1)), mg.copy()),
    breakdown("median", lambda v: float(np.median(v)), mg.copy()),
    breakdown("MAD", lambda v: float(np.median(np.abs(v - np.median(v)))), mg.copy()),
    breakdown("upper Tukey fence", lambda v: tukey(v)[4], mg.copy()),
    breakdown("10% trimmed mean", lambda v: float(stats.trim_mean(v, 0.1)), mg.copy()),
]
print("\nbreakdown, measured by moving points to 1e12 until the estimate follows:")
for r in bd:
    print(f"  {r['name']:<20} {r['points']:>4} of {N} points  ({r['fraction']:.1%})")

# ================================================================== two columns
PAIR = ["flavanoids", "total_phenols"]
A = X[:, [FEATURES.index(c) for c in PAIR]]
mu2 = A.mean(0)
S2 = np.cov(A.T, ddof=1)
S2inv = np.linalg.inv(S2)
d2_pair = np.einsum("ij,jk,ik->i", A - mu2, S2inv, A - mu2)
corr = float(np.corrcoef(A.T)[0, 1])

# Two rules, calibrated to the same nominal 97.5% so the comparison is fair:
# a rectangle from the two margins and an ellipse from the pair. A rectangle
# whose two sides are each at 97.5% covers 95.06% jointly under independence,
# so the honest matched pair is the ellipse at the same joint level.
MARG_P = 0.975
JOINT_P = MARG_P ** 2
z_cut2 = float(stats.norm.ppf(1 - (1 - MARG_P) / 2))
chi_cut2 = float(stats.chi2.ppf(JOINT_P, 2))
zc = np.abs((A - mu2) / A.std(0, ddof=1))
in_rect = (zc[:, 0] < z_cut2) & (zc[:, 1] < z_cut2)
in_ell = d2_pair < chi_cut2
rect_only = int((in_rect & ~in_ell).sum())        # margins say fine, pair says no
ell_only = int((~in_rect & in_ell).sum())         # margins say no, pair says fine
print(f"\n{PAIR[0]} against {PAIR[1]}: correlation {corr:.4f}")
print(f"  the {MARG_P:.1%} rectangle and the {JOINT_P:.2%} ellipse disagree on "
      f"{rect_only + ell_only} of the {N} bottles")
print(f"    inside both margins, outside the ellipse: {rect_only}")
print(f"    outside a margin, inside the ellipse:     {ell_only}")

# the two areas, which is where the disagreement lives whether or not a wine
# happens to sit in it. Monte Carlo over the rectangle's bounding box.
rng = np.random.default_rng(SEED)
box_lo = mu2 - z_cut2 * A.std(0, ddof=1) * 1.6
box_hi = mu2 + z_cut2 * A.std(0, ddof=1) * 1.6
Q = rng.uniform(box_lo, box_hi, size=(400000, 2))
qz = np.abs((Q - mu2) / A.std(0, ddof=1))
q_rect = (qz[:, 0] < z_cut2) & (qz[:, 1] < z_cut2)
q_ell = np.einsum("ij,jk,ik->i", Q - mu2, S2inv, Q - mu2) < chi_cut2
box_area = float(np.prod(box_hi - box_lo))
# The box has to contain both regions or the two areas below are the areas of
# whatever the box happened to keep. Checked rather than assumed: an ellipse
# tilted at this correlation is long, and the rectangle is only 1.6 times the
# marginal cutoff wide.
ell_pts = np.array([[math.cos(t), math.sin(t)] for t in np.linspace(0, 2 * math.pi, 512)])
evals2, evecs2 = np.linalg.eigh(S2)
ell_xy = mu2 + (ell_pts * np.sqrt(chi_cut2 * evals2)) @ evecs2.T
assert (ell_xy >= box_lo).all() and (ell_xy <= box_hi).all(), \
    "the sampling box does not contain the ellipse, so its area is being clipped"
assert (box_lo <= mu2 - z_cut2 * A.std(0, ddof=1)).all(), "nor the rectangle"
area = {
    "rect": rnd(q_rect.mean() * box_area, 4),
    "ellipse": rnd(q_ell.mean() * box_area, 4),
    "rect_only": rnd((q_rect & ~q_ell).mean() * box_area, 4),
    "ellipse_only": rnd((q_ell & ~q_rect).mean() * box_area, 4),
    "draws": int(Q.shape[0]),
}
area["rect_only_pct"] = rnd(100 * area["rect_only"] / area["rect"], 2)
print(f"  by area, {area['rect_only_pct']:.2f}% of the rectangle is outside the ellipse")

# the blind spot at its worst: the corner of the rectangle, which is an ordinary
# reading on each axis and an impossible one on the pair
corner = mu2 + z_cut2 * A.std(0, ddof=1) * np.array([1.0, -1.0])
corner_d2 = float((corner - mu2) @ S2inv @ (corner - mu2))
print(f"  the far corner of the rectangle is {z_cut2:.2f} sigma on each axis and "
      f"{math.sqrt(corner_d2):.2f} on the pair")

two_d = {
    "cols": PAIR, "points": [[rnd(a), rnd(b)] for a, b in A],
    "classes": [int(v) for v in Y],
    "corr": rnd(corr), "mu": [rnd(v) for v in mu2],
    "cov": [[rnd(v, 6) for v in row] for row in S2],
    "sd": [rnd(v) for v in A.std(0, ddof=1)],
    "marg_p": MARG_P, "joint_p": rnd(JOINT_P, 6),
    "z_cut": rnd(z_cut2), "chi_cut": rnd(chi_cut2),
    "d2": [rnd(v) for v in d2_pair],
    "rect_only": rect_only, "ellipse_only": ell_only,
    "rect_only_idx": [int(i) for i in np.where(in_rect & ~in_ell)[0]],
    "ellipse_only_idx": [int(i) for i in np.where(~in_rect & in_ell)[0]],
    "area": area,
    "corner": {"point": [rnd(v) for v in corner], "z": rnd(z_cut2),
               "d": rnd(math.sqrt(corner_d2))},
}

# how far the ellipse moves when the suspect is or is not in the fit: the
# widget's toggle, given a number here so the page can say what it costs
sus = np.array([float(A[:, 0].max()) * 1.02, float(A[:, 1].min()) * 0.98])
with_sus = np.vstack([A, sus])
mu_w, S_w = with_sus.mean(0), np.cov(with_sus.T, ddof=1)
d_in = float((sus - mu_w) @ np.linalg.inv(S_w) @ (sus - mu_w))
d_out = float((sus - mu2) @ S2inv @ (sus - mu2))
two_d["self_fit"] = {"point": [rnd(v) for v in sus], "d2_in": rnd(d_in),
                     "d2_out": rnd(d_out), "ratio": rnd(d_out / d_in)}
print(f"  a suspect at ({sus[0]:.2f}, {sus[1]:.2f}) scores {math.sqrt(d_in):.3f} when it is in "
      f"the fit and {math.sqrt(d_out):.3f} when it is not")

# ============================================================== thirteen columns
mu13 = X.mean(0)
S13 = np.cov(X.T, ddof=1)
S13inv = np.linalg.inv(S13)
d2_13 = np.einsum("ij,jk,ik->i", X - mu13, S13inv, X - mu13)
sum_id = float(d2_13.sum())
ceil13 = (N - 1) ** 2 / N
chi_cut13 = float(stats.chi2.ppf(0.975, P))
zmax13 = np.abs((X - mu13) / X.std(0, ddof=1)).max(1)
flag_md = set(int(i) for i in np.where(d2_13 > chi_cut13)[0])
flag_z13 = set(int(i) for i in np.where(zmax13 > 3)[0])
print(f"\nall thirteen columns:")
print(f"  the {N} squared distances sum to {sum_id:.4f}; p(n-1) = {P * (N - 1)}")
print(f"  the largest is {d2_13.max():.4f}, and none of them can pass "
      f"(n-1)^2/n = {ceil13:.4f}")
print(f"  chi-squared at 97.5% on {P} degrees of freedom is {chi_cut13:.4f}: "
      f"{len(flag_md)} bottles past it")
print(f"  screening column by column at |z| > 3 flags {len(flag_z13)}, "
      f"{len(flag_md & flag_z13)} of them the same bottles")

# ---- what the chi-squared cutoff is actually worth
# With mu and Sigma known, d^2 is chi-squared_p. Estimated from the same n
# points it is not: n*d^2/(n-1)^2 is Beta(p/2, (n-p-1)/2). Measure the gap
# instead of asserting it: draw gaussian samples, apply the textbook cutoff,
# and count how often it fires when there is nothing to find.
CAL_TRIALS = 4000
cal = []
for (nn, pp) in [(20, 2), (20, 5), (20, 13), (50, 2), (50, 5), (50, 13),
                 (178, 13), (1000, 13)]:
    if nn <= pp + 1:
        continue
    cut = float(stats.chi2.ppf(0.975, pp))
    fired = 0
    total = 0
    g = np.random.default_rng(1000 + nn * 100 + pp)
    for _ in range(CAL_TRIALS):
        Z = g.standard_normal((nn, pp))
        m = Z.mean(0)
        C = np.cov(Z.T, ddof=1) if pp > 1 else np.array([[Z.var(ddof=1)]])
        dd = np.einsum("ij,jk,ik->i", Z - m, np.linalg.inv(np.atleast_2d(C)), Z - m)
        fired += int((dd > cut).sum())
        total += nn
    exact = float(1 - stats.beta.cdf(cut * nn / (nn - 1) ** 2, pp / 2, (nn - pp - 1) / 2))
    cal.append({"n": nn, "p": pp, "nominal": 0.025, "measured": rnd(fired / total, 5),
                "exact_beta": rnd(exact, 5), "cut": rnd(cut),
                "ceiling": rnd((nn - 1) ** 2 / nn, 3),
                "reachable": bool(cut < (nn - 1) ** 2 / nn)})
print(f"\n  the textbook cutoff, checked against {CAL_TRIALS} gaussian samples that contain "
      f"nothing:")
for r in cal:
    reach = "" if r["reachable"] else "   (the cutoff is above the ceiling: it can never fire)"
    print(f"    n={r['n']:>4} p={r['p']:>2}: nominal 2.500%, measured {r['measured']:.3%}, "
          f"exact {r['exact_beta']:.3%}{reach}")

# ---- masking in thirteen dimensions: a batch that hides itself
# Plant m near-identical bottles well off the cloud and watch their own distance
# collapse as m grows, because they are writing the covariance they are measured
# against. The direction is the wine's smallest principal axis, which is where
# a contaminated batch is cheapest to hide.
evals, evecs = np.linalg.eigh(S13)
direction = evecs[:, 0]                 # smallest variance direction
scale = math.sqrt(float(evals[0]))
SHIFT = 40.0                            # in units of that direction's own sd
PLANT_MAX = 50
plant_rng = np.random.default_rng(SEED)
# Drawn once, rounded once, and shipped: the browser takes the first m of these
# rows and recomputes everything, so the widget is running the experiment rather
# than replaying it, and no random number generator has to be written twice.
PLANTED = np.round(mu13 + SHIFT * scale * direction
                   + plant_rng.standard_normal((PLANT_MAX, P)) * 0.02 * scale, DP)
mask_rows = []
for m in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 18, 25, 35, PLANT_MAX]:
    if m == 0:
        mask_rows.append({"m": 0, "d2": None, "cut": rnd(chi_cut13), "flagged": None,
                          "worst_real": rnd(float(d2_13.max())),
                          "real_flagged": int((d2_13 > chi_cut13).sum())})
        continue
    XM = np.vstack([X, PLANTED[:m]])
    mm = XM.mean(0)
    SM = np.cov(XM.T, ddof=1)
    dd = np.einsum("ij,jk,ik->i", XM - mm, np.linalg.inv(SM), XM - mm)
    cut_m = float(stats.chi2.ppf(0.975, P))
    mask_rows.append({"m": m, "d2": rnd(float(dd[N:].mean())), "cut": rnd(cut_m),
                      "flagged": int((dd[N:] > cut_m).sum()),
                      "worst_real": rnd(float(dd[:N].max())),
                      "real_flagged": int((dd[:N] > cut_m).sum())})
lost = next((r["m"] for r in mask_rows if r["m"] and r["flagged"] == 0), None)
# swamping is the other half of the same coin: while the planted batch writes
# the covariance, honest bottles cross the line that the batch itself ducks
swamp = max(mask_rows[1:], key=lambda r: r["real_flagged"])
print(f"\n  a batch planted {SHIFT:.0f} sd out along the wine's tightest direction:")
for r in mask_rows[1:]:
    print(f"    {r['m']:>3} of them: their own distance {r['d2']:>10.2f} against a cutoff of "
          f"{r['cut']:.2f}, {r['flagged']}/{r['m']} flagged")
print(f"  every one of them goes unflagged from m = {lost}" if lost
      else "  the batch is still flagged at every size tried")
print(f"  and while they hide, honest bottles get flagged instead: {swamp['real_flagged']} of the "
      f"{N} at m = {swamp['m']}, against {mask_rows[0]['real_flagged']} with no batch at all")

# the same drag in two dimensions, which is what the ellipse widget's second
# panel draws: the suspect's distance with and without itself in the fit
ray_dir = np.array([1.0, -1.0]) * A.std(0, ddof=1)
ray = []
for t in np.linspace(0, 6, 49):
    # rounded before it is scored, because the rounded one is what the page
    # will hold: measuring the unrounded point left the browser 8e-3 away from
    # a reference it reproduces exactly
    q = np.round(mu2 + t * ray_dir, DP)
    both = np.vstack([A, q])
    mb, Sb = both.mean(0), np.cov(both.T, ddof=1)
    ray.append({"t": rnd(t, 3), "point": [rnd(v) for v in q],
                "d2_in": rnd(float((q - mb) @ np.linalg.inv(Sb) @ (q - mb))),
                "d2_out": rnd(float((q - mu2) @ S2inv @ (q - mu2))),
                "z": rnd(float(np.abs((q - mu2) / A.std(0, ddof=1)).max()))})
two_d["ray"] = {"dir": [rnd(v) for v in ray_dir], "rows": ray,
                "ceiling_in": rnd((N + 1 - 1) ** 2 / (N + 1), 3)}
print(f"  dragged out along the anti-correlation ray, a suspect's own distance stops at "
      f"{ray[-1]['d2_in']:.2f} when it is in the fit and reaches {ray[-1]['d2_out']:.2f} when "
      f"it is not")

many_d = {
    "d2": [rnd(v) for v in d2_13],
    "sum": rnd(sum_id), "sum_should_be": P * (N - 1),
    "ceiling": rnd(ceil13, 3), "max": rnd(float(d2_13.max())),
    "chi_cut": rnd(chi_cut13), "p": P, "n": N,
    "flag_md": sorted(flag_md), "flag_z": sorted(flag_z13),
    "both": sorted(flag_md & flag_z13),
    "calibration": cal, "cal_trials": CAL_TRIALS,
    "masking": {"rows": mask_rows, "shift": SHIFT, "lost": lost, "swamp": swamp,
                "direction": "the smallest principal axis",
                "planted": [[rnd(v) for v in row] for row in PLANTED],
                "steps": [r["m"] for r in mask_rows]},
    "mu": [rnd(v) for v in mu13],
    "cov": [[rnd(v, 6) for v in row] for row in S13],
}

# ==================================================================== the cost
# Counted, never timed. Screening one column is a pass over n numbers; the
# ellipse needs a p by p covariance and its inverse.
cost_rows = []
for p in [1, 2, 13, 50, 200, 1000]:
    cost_rows.append({
        "p": p,
        "marginal": p * N,                                  # one pass per column
        "cov": p * (p + 1) // 2 * N,                         # the upper triangle
        "inverse": p ** 3 // 3,                              # a Cholesky solve
        "needs": p + 1,                                      # rows before S is invertible
    })
print("\ncost, counted:")
for r in cost_rows:
    print(f"  p = {r['p']:>4}: {r['marginal']:>9,} for the margins, {r['cov']:>12,} for the "
          f"covariance, and it needs {r['needs']} rows before it can be inverted")

# ================================================= references the browser checks
checks = {
    # at six decimals rather than the four the drawn arrays carry: both sides
    # start from the same rounded matrix, so the guard can be tight enough to
    # catch an implementation difference instead of measuring the print format
    "d2_pair": [rnd(v, 6) for v in d2_pair],
    "d2_13": [rnd(v, 6) for v in d2_13],
    "quantiles": {"q1": rnd(q1, 6), "q3": rnd(q3, 6)},
    "sub_stats": {"mean": rnd(float(sub.mean()), 6), "sd": rnd(float(sub.std(ddof=1)), 6),
                  "median": rnd(float(np.median(sub)), 6),
                  "mad": rnd(float(np.median(np.abs(sub - np.median(sub)))), 6)},
    "cov13_trace": rnd(float(np.trace(S13)), 6),
    "logdet13": rnd(float(np.linalg.slogdet(S13)[1]), 6),
}

out = {
    "wine": {"matrix": [[rnd(v) for v in row] for row in X],
             "classes": [int(v) for v in Y], "features": FEATURES,
             "class_names": list(wine.target_names)},
    "one_d": one_d, "breakdown": bd, "two_d": two_d, "many_d": many_d,
    "cost": {"rows": cost_rows, "n": N},
    "checks": checks,
    "meta": {"seed": SEED, "decimals": DP, "rounding_gap": raw_gap,
             "shared": shared, "sklearn": sklearn.__version__,
             "scipy": scipy.__version__, "numpy": np.__version__},
}
path = OUT / "outliers.json"
path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
print(f"\nwrote {path}  ({path.stat().st_size / 1024:.1f} kB)")
