"""The three thumbnails for the module 3.1 cards, drawn from the articles'
own measured data rather than from an illustrator's idea of them.

Each one is a picture the article actually contains, reduced to the size of a
card. A card is about 460 px wide and gets looked at for a second, so the
marks are thinned until they are separable at that size, and each file is
asserted to say what its article says before it is written: the ring thumbnail
has to show a run that really did collapse, and the sprite thumbnail has to
show a hole that is really empty.

Run from anywhere, after the three generators: `python
src/utils/generate_web_thumbnails_3_1.py`.
"""
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
            f'  <rect width="{W}" height="{H}" fill="{BG}" />',
            f'  <!-- {comment} -->']


def write(name, lines):
    lines.append('</svg>')
    path = OUT / f"{name}.svg"
    path.write_text("\n".join(lines) + "\n", "utf-8")
    print(f"  {path.name}: {path.stat().st_size / 1024:.1f} kB")


def dots(points, colour, r, opacity=0.85, per_line=6):
    """Circles grouped under one fill, which is what keeps these files small."""
    out = [f'  <g fill="{colour}" fill-opacity="{opacity}">']
    row = []
    for i, (x, y) in enumerate(points):
        row.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{r}"/>')
        if len(row) == per_line:
            out.append("    " + "".join(row))
            row = []
    if row:
        out.append("    " + "".join(row))
    out.append('  </g>')
    return out


# ----------------------------------------------------------------------------
def thumb_gans(accent="#3f6212"):
    """The ring of eight, and a run that found two of them.

    The whole article turns on the fact that the same code on a different seed
    produces this, so the thumbnail asserts it: the run drawn here has to be
    one that really lost modes.
    """
    data = json.loads((ROOT / "gans" / "data" / "ring.json").read_text("utf-8"))
    M = data["meta"]
    worst = min(((k, v) for k, v in data["runs"].items()),
                key=lambda kv: (kv[1]["final"]["covered"], kv[1]["final"]["quality"]))
    key, run = worst
    assert run["final"]["covered"] < M["k"], \
        f"nothing collapsed, so this thumbnail would be a lie: {key}"
    scale = 118 / (M["r"] * 1.35)
    to = lambda p: (W / 2 + p[0] * scale, H / 2 - p[1] * scale)

    lines = head(f"the eight modes, and the {key} run that reached "
                 f"{run['final']['covered']} of them")
    for c in M["centres"]:
        x, y = to(c)
        lines.append(f'  <circle cx="{x:.0f}" cy="{y:.0f}" r="13" fill="none" '
                     f'stroke="{INK}" stroke-width="1.4" stroke-opacity="0.55"/>')
    pts = [to(p) for p in run["points"][:170]]
    lines += dots(pts, accent, 3.6, 0.7)
    return lines


# ----------------------------------------------------------------------------
def thumb_stylegan(accent="#831843"):
    """Size against colour for the real sprites, with the rectangle the data
    never occupies drawn on top of it. The assertion is the point of the
    picture: the box has to be empty."""
    data = json.loads((ROOT / "stylegan" / "data" / "style.json").read_text("utf-8"))
    M = data["meta"]
    pts = data["scatter"]["real"]
    lo, hi = 2.0, M["size_hi"] + 2.0
    sx = lambda s: 40 + (s - lo) / (hi - lo) * (W - 70)
    sy = lambda h: H - 34 - h * (H - 66)
    # Not "exactly zero": these points are measured off pixels and the measured
    # size runs about a third of a pixel high, so a sprite drawn just under the
    # threshold lands just over it. The article publishes that share as the
    # floor; here it only has to stay small enough that the rectangle still
    # reads as empty at the size of a card.
    inside = [p for p in pts if p[0] > M["big"] and M["blue"][0] <= p[1] < M["blue"][1]]
    assert len(inside) / len(pts) < 0.02, \
        f"{len(inside)} of {len(pts)} real sprites measure inside the hole, too many to draw"

    lines = head("measured size against measured colour for the real sprites, "
                 "with the combination the data never contains")
    lines.append(f'  <rect x="{sx(M["big"]):.0f}" y="{sy(M["blue"][1]):.0f}" '
                 f'width="{W - 30 - sx(M["big"]):.0f}" '
                 f'height="{sy(M["blue"][0]) - sy(M["blue"][1]):.0f}" '
                 f'fill="{accent}" fill-opacity="0.10" stroke="{accent}" '
                 f'stroke-width="1.6" stroke-dasharray="6 4"/>')
    thin = pts[::max(1, len(pts) // 170)]
    lines += dots([(sx(p[0]), sy(p[1])) for p in thin], INK, 3.0, 0.5)
    lines.append(f'  <line x1="40" y1="{H - 34}" x2="{W - 26}" y2="{H - 34}" '
                 f'stroke="{INK}" stroke-width="1.4"/>')
    lines.append(f'  <line x1="40" y1="{H - 34}" x2="40" y2="28" '
                 f'stroke="{INK}" stroke-width="1.4"/>')
    return lines


# ----------------------------------------------------------------------------
def thumb_translation(accent="#134e4a"):
    """A plan and its render, side by side, drawn from the same colours and the
    same layout rules the scene generator uses."""
    data = json.loads((ROOT / "translation" / "data" / "translation.json").read_text("utf-8"))
    P = data["meta"]["plan_colours"]
    R = data["meta"]["render_colours"]
    plan = data["plans"]["test"][0]
    res = data["meta"]["res"]
    rgb = lambda c: "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in c)

    def panel(x0, y0, size, render):
        px = size / res
        out = [f'  <g transform="translate({x0},{y0})">']
        out.append(f'    <rect width="{size}" height="{size}" fill="{rgb(P["sky"])}"/>')
        hz = plan["horizon"] * px
        out.append(f'    <rect y="{hz:.1f}" width="{size}" height="{size - hz:.1f}" '
                   f'fill="{rgb(P["ground"])}"/>')
        ry = plan["road_y"] * px
        rh = plan["road_h"] * px
        out.append(f'    <rect y="{ry:.1f}" width="{size}" height="{rh:.1f}" '
                   f'fill="{rgb(P["road"])}"/>')
        bx = plan["bx"] * px
        bw = plan["bw"] * px
        bt = (plan["horizon"] - plan["bh"]) * px
        body = rgb(R["body"][plan["btype"]] if render else P["body"][plan["btype"]])
        out.append(f'    <rect x="{bx:.1f}" y="{bt:.1f}" width="{bw:.1f}" '
                   f'height="{hz - bt:.1f}" fill="{body}"/>')
        if render:
            if plan["btype"] == 1:
                out.append(f'    <path d="M{bx:.1f},{bt:.1f} L{bx + bw / 2:.1f},{bt - 3 * px:.1f} '
                           f'L{bx + bw:.1f},{bt:.1f} Z" fill="{body}" fill-opacity="0.8"/>')
            for k, wy in enumerate(range(plan["horizon"] - plan["bh"] + 3,
                                         plan["horizon"] - 2, 4)):
                for j, wx in enumerate(range(plan["bx"] + 2, plan["bx"] + plan["bw"] - 2, 4)):
                    on = (k + j) % 3 != 1
                    out.append(f'    <rect x="{wx * px:.1f}" y="{wy * px:.1f}" '
                               f'width="{2 * px:.1f}" height="{2 * px:.1f}" '
                               f'fill="{rgb(R["lit"] if on else R["dark"])}"/>')
            mid = (plan["road_y"] + plan["road_h"] // 2) * px
            for x in range(1, res - 1, 4):
                out.append(f'    <rect x="{x * px:.1f}" y="{mid:.1f}" width="{2 * px:.1f}" '
                           f'height="{px:.1f}" fill="#eeeee4"/>')
        out.append(f'    <rect width="{size}" height="{size}" fill="none" stroke="{INK}" '
                   f'stroke-width="1.6"/>')
        out.append('  </g>')
        return out

    size = 138
    lines = head("one plan and its render: everything in the second picture that "
                 "is not in the first is what the network has to invent")
    lines += panel(28, (H - size) / 2, size, False)
    lines += panel(W - 28 - size, (H - size) / 2, size, True)
    ax = 28 + size + 14
    bx2 = W - 28 - size - 14
    y = H / 2
    lines.append(f'  <line x1="{ax}" y1="{y}" x2="{bx2 - 8}" y2="{y}" stroke="{accent}" '
                 f'stroke-width="3"/>')
    lines.append(f'  <path d="M{bx2 - 10},{y - 7} L{bx2},{y} L{bx2 - 10},{y + 7} Z" '
                 f'fill="{accent}"/>')
    return lines


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("gans", thumb_gans), ("stylegan", thumb_stylegan),
                     ("translation", thumb_translation)):
        try:
            write(name, fn())
        except FileNotFoundError as err:
            print(f"  {name}: skipped, {err}")
