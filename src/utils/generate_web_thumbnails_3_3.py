"""The thumbnails for the module 3.3 cards, drawn from each article's own data.

Same rules as the 3.2 set: the picture is the one its article's widget draws,
thinned until the marks are separable on a 460 px card, and every number in it
comes out of that article's JSON rather than out of an idea of what the article
says.

Run from anywhere: `python src/utils/generate_web_thumbnails_3_3.py`.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "assets" / "thumbnails"
W, H = 400, 300
BG = "#f1f3f3"
INK = "#232f3e"


def head():
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
            f'  <rect width="{W}" height="{H}" fill="{BG}" />']


def write(name, lines):
    lines.append("</svg>")
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.svg"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {p.name} ({p.stat().st_size / 1024:.1f} kB)")


def load(folder, name):
    p = ROOT / folder / "data" / name
    if not p.exists():
        print(f"  (no {p.relative_to(ROOT)} yet, skipping)")
        return None
    return json.loads(p.read_text("utf-8"))


# ------------------------------------------------------------------------ ddpm
# The article's subject is the forward process, so the card is the forward
# process: the same density at four noise levels, left to right, dissolving.
d = load("ddpm", "ddpm.json")
if d:
    ACC = "#674599"
    rows = d["destroy"]["rows"]
    pick = [rows[0], rows[4], rows[8], rows[-1]]
    lines = head()
    lines.append("  <!-- the truth at four noise levels, and what is left of it -->")
    panel = 84
    gap = 12
    x0 = (W - (4 * panel + 3 * gap)) / 2
    # the block runs from the t labels down to the caption, so it is centred
    # against that height rather than against the panels alone
    top = 46 + 35
    # Not a point cloud: at card size 900 dots is 68 kB of SVG and an
    # illegible smudge. Each component is drawn as the shape it is, with the
    # blur taken from the schedule rather than chosen. The ring's radius
    # shrinks by sqrt(abar) and its stroke widens by sqrt(1 - abar), which is
    # exactly what the forward process does to it.
    for k, r in enumerate(pick):
        a = float(r["abar"])
        sa, sd = np.sqrt(a), np.sqrt(max(1 - a, 0.0))
        x = x0 + k * (panel + gap)
        cx, cy = x + panel / 2, top + panel / 2
        u = panel / 9.2                      # pixels per unit of the plane
        lines.append(f'  <rect x="{x:.0f}" y="{top}" width="{panel}" height="{panel}" '
                     f'fill="#ffffff" stroke="#c9d1d6" stroke-width="1" />')
        ring_r = 2.0 * sa * u
        ring_w = max(1.2, 2.0 * np.hypot(0.16 * sa, sd) * u)
        # sqrt(1 - abar) never exceeds one, so there is no faded branch here:
        # the blur is carried by the shapes' own size, not by opacity
        op = 0.85
        lines.append(f'    <circle cx="{cx:.1f}" cy="{cy:.1f}" r="{ring_r:.1f}" '
                     f'fill="none" stroke="{ACC}" stroke-width="{ring_w:.1f}" '
                     f'opacity="{op:.2f}" />')
        for (mx, my), (s1, s2), rot in ((( -2.55, 2.55), (0.35, 0.35), 0),
                                        ((2.55, -2.55), (0.62, 0.17), -35)):
            ex = np.hypot(s1 * sa, sd) * u
            ey = np.hypot(s2 * sa, sd) * u
            bx = cx + mx * sa * u
            by = cy - my * sa * u
            lines.append(f'    <ellipse cx="{bx:.1f}" cy="{by:.1f}" rx="{ex:.1f}" '
                         f'ry="{ey:.1f}" fill="{ACC}" opacity="{op * 0.7:.2f}" '
                         f'transform="rotate({rot} {bx:.1f} {by:.1f})" />')
        lines.append(f'  <text x="{cx:.0f}" y="{top - 10}" text-anchor="middle" '
                     f'font-family="monospace" font-size="12" fill="{INK}">t={r["t"]}</text>')
        lines.append(f'  <text x="{cx:.0f}" y="{top + panel + 16}" '
                     f'text-anchor="middle" font-family="monospace" font-size="11" '
                     f'fill="{INK}" opacity="0.75">{r["kl_normal"]}</text>')
    lines.append(f'  <text x="{W / 2:.0f}" y="{top + panel + 44}" text-anchor="middle" '
                 f'font-family="monospace" font-size="12" fill="{INK}">'
                 f'nats from a standard normal, exactly</text>')
    write("ddpm", lines)


# --------------------------------------------------------------- latent-diffusion
# The article is about two costs pulling opposite ways, so the card is that
# chart with everything stripped but the crossing.
d = load("latent-diffusion", "latent.json")
if d:
    ACC = "#995e45"
    T = d["trade"]
    lines = head()
    lines.append("  <!-- the ceiling against what the diffusion reaches -->")
    L, R, TOP, BOT = 66, W - 34, 70, 232
    ks = [r["k"] for r in T]
    vals = [abs(float(r["mmd2"])) for r in T] + [abs(float(r["ceiling_mmd2"])) for r in T]
    lo, hi = np.log10(min(vals)), np.log10(max(vals))
    px = lambda k: L + (R - L) * (np.log2(k) - np.log2(ks[0])) / (np.log2(ks[-1]) - np.log2(ks[0]))
    py = lambda v: BOT - (BOT - TOP) * (np.log10(abs(float(v))) - lo) / (hi - lo)
    for key, colour, dash, name in (("ceiling_mmd2", "#232f3e", ' stroke-dasharray="6 4"',
                                     "what the decoder allows"),
                                    ("mmd2", ACC, "", "what the diffusion reaches")):
        pts = " ".join(f"{px(r['k']):.0f},{py(r[key]):.0f}" for r in T)
        lines.append(f'  <polyline points="{pts}" fill="none" stroke="{colour}" '
                     f'stroke-width="2.4"{dash} />')
        for r in T:
            lines.append(f'    <circle cx="{px(r["k"]):.0f}" cy="{py(r[key]):.0f}" r="4" '
                         f'fill="{colour}" />')
        lines.append(f'  <text x="{R}" y="{py(T[-1][key]) + (16 if key == "ceiling_mmd2" else -10):.0f}" '
                     f'text-anchor="end" font-family="monospace" font-size="12" '
                     f'fill="{colour}">{name}</text>')
    for r in T:
        lines.append(f'  <text x="{px(r["k"]):.0f}" y="{BOT + 20}" text-anchor="middle" '
                     f'font-family="monospace" font-size="12" fill="{INK}">{r["k"]}</text>')
    lines.append(f'  <text x="{(L + R) / 2:.0f}" y="{BOT + 44}" text-anchor="middle" '
                 f'font-family="monospace" font-size="12" fill="{INK}">'
                 f'numbers in the bottleneck</text>')
    lines.append(f'  <text x="{(L + R) / 2:.0f}" y="{TOP - 28}" text-anchor="middle" '
                 f'font-family="monospace" font-size="12" fill="{INK}">'
                 f'the ceiling, and what reaches it</text>')
    write("latent-diffusion", lines)



# ------------------------------------------------------------------- controlnet
# The card is the finding: two curves against the same dial, peaking apart.
d = load("controlnet", "control.json")
if d:
    ACC = "#994573"
    S = d["sweep"]
    lines = head()
    lines.append("  <!-- obedience and realism against the guidance scale -->")
    L, R, TOP, BOT = 58, W - 40, 74, 226
    xs = [r["scale"] for r in S]
    px = lambda v: L + (R - L) * (v - xs[0]) / (xs[-1] - xs[0])
    obey = [r["obeys"] for r in S]
    mm = [abs(float(r["mmd2"])) for r in S]
    pyo = lambda v: BOT - (BOT - TOP) * v / 1.05
    lo, hi = np.log10(min(mm)), np.log10(max(mm))
    pym = lambda v: BOT - (BOT - TOP) * (np.log10(abs(float(v))) - lo) / (hi - lo)
    # the judge on real digits, which is the line the obedience curve passes
    lines.append(f'  <line x1="{L}" y1="{pyo(d["real"]["obeys"]):.0f}" x2="{R}" '
                 f'y2="{pyo(d["real"]["obeys"]):.0f}" stroke="#1d4ed8" stroke-width="1.6" '
                 f'stroke-dasharray="6 4" />')
    for key, fy, colour, dash, name in (("obeys", pyo, ACC, "", "obeys"),
                                        ("mmd2", pym, "#232f3e", ' stroke-dasharray="6 4"',
                                         "looks real")):
        pts = " ".join(f"{px(r['scale']):.0f},{fy(r[key]):.0f}" for r in S)
        lines.append(f'  <polyline points="{pts}" fill="none" stroke="{colour}" '
                     f'stroke-width="2.4"{dash} />')
        for r in S:
            lines.append(f'    <circle cx="{px(r["scale"]):.0f}" cy="{fy(r[key]):.0f}" r="3.6" '
                         f'fill="{colour}" />')
    bo = max(S, key=lambda r: r["obeys"])
    br = min(S, key=lambda r: abs(float(r["mmd2"])))
    for r, fy, colour, key in ((bo, pyo, ACC, "obeys"), (br, pym, "#232f3e", "mmd2")):
        lines.append(f'  <line x1="{px(r["scale"]):.0f}" y1="{fy(r[key]):.0f}" '
                     f'x2="{px(r["scale"]):.0f}" y2="{BOT}" stroke="{colour}" stroke-width="1.2" '
                     f'stroke-dasharray="3 3" />')
    lines.append(f'  <text x="{px(bo["scale"]):.0f}" y="{BOT + 20}" text-anchor="middle" '
                 f'font-family="monospace" font-size="12" fill="{ACC}">obeys best: {bo["scale"]}</text>')
    lines.append(f'  <text x="{px(br["scale"]):.0f}" y="{TOP - 12}" text-anchor="middle" '
                 f'font-family="monospace" font-size="12" fill="#232f3e">looks best: {br["scale"]}</text>')
    lines.append(f'  <text x="{(L + R) / 2:.0f}" y="{BOT + 44}" text-anchor="middle" '
                 f'font-family="monospace" font-size="12" fill="{INK}">guidance scale</text>')
    lines.append(f'  <text x="{(L + R) / 2:.0f}" y="{TOP - 34}" text-anchor="middle" '
                 f'font-family="monospace" font-size="12" fill="{INK}">'
                 f'the two peaks are not the same</text>')
    write("controlnet", lines)



# ------------------------------------------------------------------ consistency
# The card is the gap at the left edge: the same model, one evaluation, before
# and after its paths were straightened. The two curves meet at the right, which
# is the other half of the point, so both ends are worth drawing.
d = load("consistency", "consistency.json")
if d:
    ACC = "#5c293b"
    A = d["arms"]
    B = d["meta"]["budgets"]
    rows = lambda name: sorted((r for r in A if r["arm"] == name), key=lambda r: r["steps"])
    lines = head()
    lines.append("  <!-- distance from real digits against the step budget -->")
    L, R, TOP, BOT = 60, W - 34, 78, 222
    vals = [abs(float(r["mmd2"])) for r in A] + [abs(float(d["setup"]["floor"]["mmd2_max"]))]
    lo, hi = np.log10(min(vals)), np.log10(max(vals))
    px = lambda s: L + (R - L) * (np.log10(s) - np.log10(B[0])) / (np.log10(B[-1]) - np.log10(B[0]))
    py = lambda v: BOT - (BOT - TOP) * (np.log10(abs(float(v))) - lo) / (hi - lo)
    floor_y = py(d["setup"]["floor"]["mmd2_max"])
    lines.append(f'  <line x1="{L}" y1="{floor_y:.0f}" x2="{R}" y2="{floor_y:.0f}" '
                 f'stroke="#1d4ed8" stroke-width="1.4" stroke-dasharray="6 4" opacity="0.8" />')
    lines.append(f'  <text x="{R}" y="{floor_y + 15:.0f}" text-anchor="end" '
                 f'font-family="monospace" font-size="11" fill="#1d4ed8">real against real</text>')
    for name, colour, dash in (("rectified flow", "#232f3e", ' stroke-dasharray="6 4"'),
                               ("after reflow", ACC, "")):
        rs = rows(name)
        pts = " ".join(f"{px(r['steps']):.0f},{py(r['mmd2']):.0f}" for r in rs)
        lines.append(f'  <polyline points="{pts}" fill="none" stroke="{colour}" '
                     f'stroke-width="2.4"{dash} />')
        for r in rs:
            lines.append(f'    <circle cx="{px(r["steps"]):.0f}" cy="{py(r["mmd2"]):.0f}" r="3.6" '
                         f'fill="{colour}" />')
    a1 = rows("rectified flow")[0]
    b1 = rows("after reflow")[0]
    ya, yb = py(a1["mmd2"]), py(b1["mmd2"])
    ratio = abs(float(a1["mmd2"])) / abs(float(b1["mmd2"]))
    # The gap at one evaluation, drawn as the bracket it is. The label does NOT
    # go beside the bracket: the falling curve passes straight through that
    # space, and the first draft put a caption on top of its own data. It goes
    # in the empty band between the flat curve and the floor line, which is the
    # only region of this card with nothing in it.
    lines.append(f'  <line x1="{L + 12}" y1="{ya:.0f}" x2="{L + 12}" y2="{yb:.0f}" '
                 f'stroke="{ACC}" stroke-width="1.4" stroke-dasharray="3 3" />')
    for yy in (ya, yb):
        lines.append(f'    <line x1="{L + 7}" y1="{yy:.0f}" x2="{L + 17}" y2="{yy:.0f}" '
                     f'stroke="{ACC}" stroke-width="1.4" />')
    lines.append(f'  <text x="{L + 18}" y="{BOT - 12:.0f}" text-anchor="start" '
                 f'font-family="monospace" font-size="12" fill="{ACC}">'
                 f'{ratio:.0f} times closer at one evaluation</text>')
    lines.append(f'  <text x="{px(1):.0f}" y="{ya - 12:.0f}" text-anchor="start" '
                 f'font-family="monospace" font-size="11" fill="#232f3e">rectified flow</text>')
    lines.append(f'  <text x="{px(B[-1]):.0f}" y="{py(rows("after reflow")[-1]["mmd2"]) + 18:.0f}" '
                 f'text-anchor="end" font-family="monospace" font-size="11" '
                 f'fill="{ACC}">after reflow</text>')
    for s in (1, 10, 100):
        lines.append(f'  <text x="{px(s):.0f}" y="{BOT + 20}" text-anchor="middle" '
                     f'font-family="monospace" font-size="12" fill="{INK}">{s}</text>')
    lines.append(f'  <text x="{(L + R) / 2:.0f}" y="{BOT + 44}" text-anchor="middle" '
                 f'font-family="monospace" font-size="12" fill="{INK}">'
                 f'network evaluations per sample</text>')
    lines.append(f'  <text x="{(L + R) / 2:.0f}" y="{TOP - 34}" text-anchor="middle" '
                 f'font-family="monospace" font-size="12" fill="{INK}">'
                 f'one evaluation, before and after</text>')
    write("consistency", lines)


print("done")
