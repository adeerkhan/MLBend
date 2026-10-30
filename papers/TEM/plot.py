"""The TEM paper figures: curves and samples, in the bend-lang palette.

  reference/.venv-lucidrains/bin/python papers/TEM/plot.py --tag tem64-pra
"""
import argparse
import base64
import json
import os
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results")
BG = "#f2eee7"
GRID = "#d8d2c6"
INK = "#4d4a44"
MUTED = "#87847d"
VIO = "#8b83b5"
VIO_D = "#5e5787"
GREEN = "#7e9a5e"
ORANGE = "#c4845c"
RED = "#c46a60"
SANS = "Inter, 'Segoe UI', system-ui, sans-serif"
MONO = "ui-monospace, 'SFMono-Regular', Menlo, monospace"


def text(x, y, s, size=14, fill=INK, anchor="start", weight="400", fam=SANS):
    return (f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" '
            f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">'
            f'{s}</text>')


def line(x1, y1, x2, y2, stroke=GRID, width=1):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
            f'stroke-width="{width}"/>')


def frame(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}"><rect width="{w}" height="{h}" fill="{BG}"/>'
            f'{body}</svg>')


def save(name, svg):
    p = os.path.join(OUT, name)
    open(p + ".svg", "w").write(svg)
    subprocess.run(["convert", "-density", "192", p + ".svg", p + ".webp"],
                   check=True, capture_output=True)
    print("wrote", p + ".webp")


def curves(tag):
    rec = json.load(open(os.path.join(OUT, tag + ".json")))
    c = rec["curve"]
    steps = [p["step"] for p in c]
    W, H = 1200, 560
    L, R, T = 90, 40, 120
    body = [text(60, 56, "TEM-small on Oxford Flowers 64x64", 26, INK, weight="600"),
            text(60, 86, f"{rec['steps']} steps, CD + L2 + PRA, batch {rec['batch']}, "
                         f"{rec['params'] / 1e6:.2f}M parameters", 14, MUTED)]
    # top panel: total, CD and PRA
    panes = [("loss and terms", [("total", "loss", VIO), ("CD", "cd", ORANGE),
                                 ("PRA", "pra", GREEN)], T, 200),
             ("energy", [("positive", "pos", VIO_D), ("negative", "neg", RED)],
              T + 250, 200)]
    for title, series, top, ph in panes:
        vals = [pt[k] for _, k, _ in series for pt in c if k in pt]
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-3:
            hi = lo + 1
        pad = 0.08 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        body.append(text(L, top - 12, title, 13, MUTED, weight="600"))
        for t in (0.0, 0.5, 1.0):
            v = lo + t * (hi - lo)
            y = top + ph - t * ph
            body.append(line(L, y, L + 820, y))
            body.append(text(L - 12, y + 4, f"{v:.2f}", 11, MUTED, anchor="end",
                             fam=MONO))
        for name, key, color in series:
            if key not in c[0]:
                continue
            pts = []
            for p in c:
                x = L + (p["step"] / max(1, steps[-1])) * 820
                y = top + ph - (p[key] - lo) / (hi - lo) * ph
                pts.append(f"{x:.1f},{y:.1f}")
            body.append(f'<polyline points="{" ".join(pts)}" fill="none" '
                        f'stroke="{color}" stroke-width="2.2"/>')
            body.append(text(L + 830, top + 10 + 18 * series.index(
                (name, key, color)), name, 12, color, fam=MONO))
    body.append(text(L + 410, H - 18, "step", 12, MUTED, anchor="middle"))
    save(f"{tag}-curves", frame(W, H, "".join(body)))


def samples(tag):
    png = os.path.join(OUT, tag + "-samples.png")
    b64 = base64.b64encode(open(png, "rb").read()).decode()
    W, H = 1200, 760
    body = [text(60, 56, "TEM-small samples", 26, INK, weight="600"),
            text(60, 86, "64 labels from the flowers test set, 20 Langevin steps each",
                 14, MUTED),
            f'<image x="300" y="130" width="600" height="600" '
            f'href="data:image/png;base64,{b64}"/>']
    save(f"{tag}-samples-fig", frame(W, H, "".join(body)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="tem64-pra")
    a = ap.parse_args()
    curves(a.tag)
    samples(a.tag)
