"""The seven thumbnails for the module 4.2 and 4.3 cards.

Each one is drawn from the article's own measured file, not from an idea of
what the article is about: the carbon dioxide curve with the fan the seasonal
ARIMA actually produced, the volatility path the fitted GARCH actually took, the
feature matrix the boosted tree was actually handed, and so on.

A card is 460 px wide and gets looked at for a second, so everything here is
thinned until the marks separate at that size, coordinates are rounded to
integers, and marks of one colour are grouped so the fill is written once. That
keeps these files in the 2-8 kB band the other thirty-odd thumbnails live in.

Run from anywhere: `python src/utils/generate_web_thumbnails_4.py`.
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
GREY = "#8e9aa6"


def head():
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
            f'  <rect width="{W}" height="{H}" fill="{BG}" />']


def write(name, parts):
    parts.append("</svg>")
    path = OUT / f"{name}.svg"
    path.write_text("\n".join(parts) + "\n")
    print(f"  {path.name}: {path.stat().st_size / 1024:.1f} kB")


def scale(vals, lo=None, hi=None, top=20, bottom=H - 20):
    v = np.asarray(vals, dtype=float)
    lo = v.min() if lo is None else lo
    hi = v.max() if hi is None else hi
    if hi <= lo:
        hi = lo + 1
    return bottom - (v - lo) / (hi - lo) * (bottom - top)


def polyline(xs, ys, colour, width=2, opacity=None):
    pts = " ".join(f"{x:.0f},{y:.0f}" for x, y in zip(xs, ys))
    op = "" if opacity is None else f' opacity="{opacity}"'
    return (f'  <polyline points="{pts}" fill="none" stroke="{colour}" '
            f'stroke-width="{width}"{op} />')


def load(article, name):
    return json.loads((ROOT / article / "data" / f"{name}.json").read_text())


# ---------------------------------------------------------------------------
def arima_thumb():
    d = load("arima", "arima")
    demo = d["rolling"]["demo"]
    hist = demo["history"][-40:]
    fan_lo = demo["lo95"]
    fan_hi = demo["hi95"]
    mid = demo["sarima"]
    n = len(hist)
    allv = hist + fan_lo + fan_hi
    ys = scale(allv)
    yh, ylo, yhi = ys[:n], ys[n:n + len(fan_lo)], ys[n + len(fan_lo):]
    xs = np.linspace(24, W * 0.62, n)
    xf = np.linspace(W * 0.62, W - 16, len(fan_lo))
    ym = scale(mid, min(allv), max(allv))
    parts = head()
    band = (" ".join(f"{x:.0f},{y:.0f}" for x, y in zip(xf, ylo))
            + " " + " ".join(f"{x:.0f},{y:.0f}" for x, y in zip(xf[::-1], yhi[::-1])))
    parts.append(f'  <polygon points="{band}" fill="#713f12" opacity="0.22" />')
    parts.append(polyline(xs, yh, INK, 2))
    parts.append(polyline(xf, ym, "#713f12", 2.6))
    write("arima", parts)


def volatility_thumb():
    d = load("volatility", "volatility")
    vol = np.array(d["garch"]["vol"])
    step = max(1, len(vol) // 150)
    v = vol[::step] * np.sqrt(252)
    ys = scale(v, 0, float(v.max()) * 1.05)
    xs = np.linspace(16, W - 16, len(v))
    ret = np.abs(np.array(d["returns"]["y"])[::step][:len(v)]) * np.sqrt(252)
    yr = scale(ret, 0, float(v.max()) * 1.05)
    parts = head()
    # the days as one filled path rather than 150 lines: a third of the bytes
    poly = " ".join(f"{x:.0f},{y:.0f}" for x, y in zip(xs, yr))
    parts.append(f'  <polygon points="{xs[0]:.0f},{H - 20} {poly} {xs[-1]:.0f},{H - 20}" '
                 f'fill="{GREY}" opacity="0.45" />')
    parts.append(polyline(xs, ys, "#851463", 2.4))
    write("volatility", parts)


def timefeatures_thumb():
    d = load("time-features", "timefeatures")
    y = np.array(d["series"]["y"])
    rows, cols = 14, 9
    end = len(y) - 1
    mat = np.zeros((rows, cols))
    for i in range(rows):
        t = end - (rows - 1 - i)
        mat[i] = [y[t - 1], y[t - 2], y[t - 3], y[t - 12],
                  np.sin(2 * np.pi * (t % 12) / 12), np.cos(2 * np.pi * (t % 12) / 12),
                  y[t - 12:t].mean(), y[t - 1] - y[t - 2], y[t]]
    parts = head()
    cw, ch = 30, 16
    x0, y0 = 46, 44
    shades = {}
    for j in range(cols):
        col = mat[:, j]
        lo, hi = col.min(), col.max()
        for i in range(rows):
            v = 0.5 if hi <= lo else (col[i] - lo) / (hi - lo)
            q = (j == cols - 1, int(v * 5 + 0.5))
            shades.setdefault(q, []).append(
                f'<rect x="{x0 + j * cw}" y="{y0 + i * ch}" width="{cw - 2}" '
                f'height="{ch - 2}" />')
    for (target, q), items in sorted(shades.items()):
        v = q / 5
        if target:
            c = int(247 - v * (247 - 160)), int(223 - v * (223 - 32)), int(226 - v * (226 - 64))
        else:
            c = int(238 - v * (238 - 133)), int(240 - v * (240 - 60)), int(240 - v * (240 - 67))
        parts.append(f'  <g fill="rgb({c[0]},{c[1]},{c[2]})">')
        for i in range(0, len(items), 7):
            parts.append("    " + "".join(items[i:i + 7]))
        parts.append("  </g>")
    parts.append(f'  <text x="{x0 + (cols - 1) * cw + cw / 2 - 1}" y="{y0 - 10}" '
                 f'text-anchor="middle" font-family="monospace" font-size="13" fill="{INK}">y</text>')
    parts.append(f'  <text x="{x0 - 8}" y="{y0 + rows * ch / 2}" text-anchor="end" '
                 f'font-family="monospace" font-size="12" fill="{INK}">rows</text>')
    write("time-features", parts)


def deepforecast_thumb():
    d = load("deep-forecast", "deepforecast")
    S = d["scaling"]
    xs_v = np.log10([r["series"] for r in S["rows"]])
    vals = [r["mase"] for r in S["rows"]] + [S["local_nbeats"], S["ets"], S["seasonal_naive"]]
    lo, hi = min(vals) * 0.95, max(vals) * 1.05
    xs = 40 + (xs_v - xs_v.min()) / (xs_v.max() - xs_v.min()) * (W - 80)
    ys = scale([r["mase"] for r in S["rows"]], lo, hi, top=34, bottom=H - 34)
    parts = head()
    for v, colour in [(S["local_nbeats"], GREY), (S["ets"], GREY), (S["seasonal_naive"], GREY)]:
        yv = scale([v], lo, hi, top=34, bottom=H - 34)[0]
        parts.append(f'  <line x1="24" y1="{yv:.0f}" x2="{W - 24}" y2="{yv:.0f}" '
                     f'stroke="{GREY}" stroke-width="1.4" stroke-dasharray="6 4" />')
    parts.append(polyline(xs, ys, "#3c4e85", 3))
    circles = "".join(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="5" />' for x, y in zip(xs, ys))
    parts.append(f'  <g fill="#3c4e85">{circles}</g>')
    write("deep-forecast", parts)


def zeroshot_thumb():
    d = load("zero-shot", "zeroshot")
    D = d["quantisation"]["demo"]
    win = np.array(D["window"])
    back = np.array(D["reconstructed"])
    lo, hi = min(win.min(), back.min()) - 0.5, max(win.max(), back.max()) + 0.5
    xs = np.linspace(20, W - 20, len(win))
    yw = scale(win, lo, hi)
    yb = scale(back, lo, hi)
    parts = head()
    parts.append(polyline(xs, yw, GREY, 1.6))
    steps_x, steps_y = [], []
    for i in range(len(xs)):
        steps_x.append(xs[i])
        steps_y.append(yb[i])
        if i + 1 < len(xs):
            steps_x.append(xs[i + 1])
            steps_y.append(yb[i])
    parts.append(polyline(steps_x, steps_y, "#0e5c29", 2.6))
    write("zero-shot", parts)


def spectrograms_thumb():
    d = load("spectrograms", "spectrograms")
    db = np.array(d["chirp"]["spec"]["db"])
    db = db[:, :96]
    cols = min(46, db.shape[0])
    idx = np.linspace(0, db.shape[0] - 1, cols).astype(int)
    rows = 26
    ridx = np.linspace(0, db.shape[1] - 1, rows).astype(int)
    m = db[np.ix_(idx, ridx)]
    lo, hi = m.max() - 45, m.max()
    cw = (W - 32) / cols
    ch = (H - 40) / rows
    parts = head()
    shades = {}
    for ci in range(cols):
        for ri in range(rows):
            v = float(np.clip((m[ci, ri] - lo) / (hi - lo), 0, 1))
            if v < 0.12:
                continue
            # quantise the ramp to eight shades and group the rects that share
            # one, so the fill is written eight times instead of a thousand
            q = int(v * 7 + 0.5)
            shades.setdefault(q, []).append(
                f'<rect x="{16 + ci * cw:.0f}" y="{H - 20 - (ri + 1) * ch:.0f}" '
                f'width="{cw + 1:.0f}" height="{ch + 1:.0f}" />')
    for q in sorted(shades):
        v = q / 7
        c = (int(241 - v * (241 - 92)), int(243 - v * (243 - 14)), int(243 - v * (243 - 29)))
        parts.append(f'  <g fill="rgb({c[0]},{c[1]},{c[2]})">')
        for i in range(0, len(shades[q]), 6):
            parts.append("    " + "".join(shades[q][i:i + 6]))
        parts.append("  </g>")
    write("spectrograms", parts)


def wavenet_thumb():
    layers = 5
    n = 33
    parts = head()
    x = lambda i: 20 + (W - 40) * i / (n - 1)  # noqa: E731
    yof = lambda l: H - 30 - l * ((H - 70) / layers)  # noqa: E731
    reach = [n - 1]
    lines = []
    for l in range(1, layers + 1):
        d = 2 ** (l - 1)
        nxt = []
        for p in reach:
            for q in (p, p - d):
                if q >= 0 and q not in nxt:
                    nxt.append(q)
                if q >= 0:
                    lines.append((x(p), yof(l - 1), x(q), yof(l)))
        reach = nxt
    parts.append('  <g stroke="#5c3e29" stroke-width="1.1" opacity="0.55">')
    for i in range(0, len(lines), 4):
        parts.append("    " + "".join(
            f'<line x1="{a:.0f}" y1="{b:.0f}" x2="{c:.0f}" y2="{e:.0f}" />'
            for a, b, c, e in lines[i:i + 4]))
    parts.append("  </g>")
    dots = []
    for l in range(layers + 1):
        for i in range(0, n, 2):
            dots.append(f'<circle cx="{x(i):.0f}" cy="{yof(l):.0f}" r="2.4" />')
    parts.append(f'  <g fill="{GREY}" opacity="0.5">')
    for i in range(0, len(dots), 10):
        parts.append("    " + "".join(dots[i:i + 10]))
    parts.append("  </g>")
    parts.append(f'  <g fill="#5c3e29"><circle cx="{x(n - 1):.0f}" cy="{yof(layers):.0f}" '
                 f'r="5" /></g>')
    write("wavenet", parts)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in [("arima", arima_thumb), ("volatility", volatility_thumb),
                     ("time-features", timefeatures_thumb),
                     ("deep-forecast", deepforecast_thumb), ("zero-shot", zeroshot_thumb),
                     ("spectrograms", spectrograms_thumb), ("wavenet", wavenet_thumb)]:
        try:
            fn()
        except FileNotFoundError as exc:
            print(f"  {name}: skipped, {exc}")


if __name__ == "__main__":
    main()
