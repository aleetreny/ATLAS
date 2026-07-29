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
    ACC = "#164e63"
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

print("done")
