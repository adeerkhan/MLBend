"""The README figures, from the result JSONs, in the bend-lang palette.

  reference/.venv/bin/python reference/make_readme_figures.py

Writes docs/figures/{legibility,loss,flow,plumbing}.svg and .webp, and an
index.html gallery. The webp is what the README embeds; the svg is the source.
The numbers come from results/, the before/after plumbing numbers from the
2026-10-30 measurement recorded in ISSUE-018.
"""
import json
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "figures")

BG = "#f2eee7"
PANEL = "#e9e4db"
GRID = "#d8d2c6"
INK = "#4d4a44"
MUTED = "#87847d"
VIO = "#8b83b5"
VIO_D = "#5e5787"
VIO_L = "#d2cee1"
GREEN = "#7e9a5e"
ORANGE = "#c4845c"
RED = "#c46a60"
SANS = "Inter, 'Segoe UI', system-ui, sans-serif"
MONO = "ui-monospace, 'SFMono-Regular', Menlo, monospace"


def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def text(x, y, s, size=14, fill=INK, anchor="start", weight="400", family=SANS,
         opacity=1.0):
    return (f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
            f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}" '
            f'opacity="{opacity}">{esc(s)}</text>')


def rect(x, y, w, h, fill, rx=0, opacity=1.0):
    return (f'<rect x="{x}" y="{y}" width="{max(0, w)}" height="{h}" rx="{rx}" '
            f'fill="{fill}" opacity="{opacity}"/>')


def line(x1, y1, x2, y2, stroke=GRID, width=1, dash=None, opacity=1.0):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
            f'stroke-width="{width}" opacity="{opacity}"{d}/>')


def frame(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}">{rect(0, 0, w, h, BG)}{body}</svg>')


def save(name, svg):
    os.makedirs(OUT, exist_ok=True)
    svg_path = os.path.join(OUT, name + ".svg")
    webp_path = os.path.join(OUT, name + ".webp")
    with open(svg_path, "w") as f:
        f.write(svg)
    subprocess.run(["convert", "-density", "192", svg_path, webp_path],
                   check=True, capture_output=True)
    print("wrote", os.path.relpath(webp_path, ROOT))


def card_title(body, title, sub):
    body.append(text(60, 56, title, 26, INK, weight="600"))
    body.append(text(60, 86, sub, 14, MUTED))


# ---- legibility by arm -------------------------------------------------------

def legibility():
    compare = json.load(open(os.path.join(ROOT, "results/compare/compare.json")))
    arms = [("Bend-ViT", "bend-vit", VIO),
            ("PyTorch", "torch-vit", GREEN),
            ("lucidrains UNet", "lucidrains-unet", ORANGE)]
    body = []
    card_title(body, "Legibility on 8×8 digits",
               "mean best absolute correlation of a sample with any training digit, 4096 samples")
    L, R, T, B = 230, 50, 130, 80
    W, H = 1200, 620
    pw, ph = W - L - R, H - T - B
    y0 = T + ph

    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = y0 - tick * ph
        body.append(line(L, y, L + pw, y, GRID, 1))
        body.append(text(L - 14, y + 5, f"{tick:.2f}", 13, MUTED, anchor="end",
                         family=MONO))
    gate_y = y0 - 0.75 * ph
    body.append(line(L, gate_y, L + pw, gate_y, RED, 1.5, "6 5", 0.85))
    body.append(text(L + pw, gate_y - 8, "gate 0.75", 12, RED, anchor="end"))

    gw = pw / len(arms)
    bar_w, gap = 78, 14
    for i, (label, key, color) in enumerate(arms):
        legs = compare["arms"][key]["samplers"]
        cx = L + gw * i + gw / 2
        body.append(text(cx, y0 + 34, label, 15, INK, anchor="middle",
                         weight="600"))
        for j, (sampler, name, shade, op) in enumerate(
                (("ddim", "DDIM 16", color, 1.0),
                 ("ddpm", "ancestral 64", color, 0.55))):
            v = legs[sampler]["legibility"]
            bx = cx - bar_w - gap / 2 + j * (bar_w + gap)
            bh = v * ph
            body.append(rect(bx, y0 - bh, bar_w, bh, shade, 3, op))
            body.append(text(bx + bar_w / 2, y0 - bh - 10, f"{v:.3f}", 14, INK,
                             anchor="middle", weight="600", family=MONO))
    body.append(text(L + 8, 118, "solid: DDIM 16    light: ancestral 64", 12,
                     MUTED))
    save("legibility", frame(W, H, "".join(body)))


# ---- the ViT loss curve ------------------------------------------------------

def loss():
    run = json.load(open(os.path.join(ROOT, "results/diffusion-vit/run.json")))
    curve = run["curve"]
    steps = run["steps"]
    body = []
    card_title(body, "ViT denoising loss",
               f"{steps} steps, MSE on 1024 optdigits, 100,736 parameters, causal attention")
    L, R, T, B = 90, 380, 130, 72
    W, H = 1200, 540
    pw, ph = W - L - R, H - T - B
    y0 = T + ph
    lo, hi = 0.05, 1.2

    def yv(v):
        import math
        t = (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo))
        return y0 - t * ph

    for tick in (0.05, 0.1, 0.25, 0.5, 1.0):
        y = yv(tick)
        body.append(line(L, y, L + pw, y, GRID, 1))
        body.append(text(L - 14, y + 5, f"{tick:g}", 13, MUTED, anchor="end",
                         family=MONO))
    n = len(curve) - 1
    pts = [f"{L + (i / n) * pw:.1f},{yv(v):.1f}" for i, v in enumerate(curve)]
    body.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{VIO}" '
                f'stroke-width="2.6" stroke-linejoin="round"/>')
    body.append(f'<circle cx="{L:.1f}" cy="{yv(curve[0]):.1f}" r="4" fill="{VIO_D}"/>')
    mi = min(range(len(curve)), key=lambda i: curve[i])
    mx = L + (mi / n) * pw
    body.append(f'<circle cx="{mx:.1f}" cy="{yv(curve[mi]):.1f}" r="4" fill="{VIO_D}"/>')
    body.append(text(mx + 8, yv(curve[mi]) - 12, f"min {curve[mi]:.3f}", 13, VIO_D,
                     family=MONO))
    body.append(text(L + pw / 2, y0 + 44, "training step", 13, MUTED,
                     anchor="middle"))

    px = L + pw + 70
    body.append(text(px, T + 24, "first", 13, MUTED))
    body.append(text(px, T + 52, f"{curve[0]:.3f}", 22, INK, family=MONO))
    body.append(text(px, T + 100, "final", 13, MUTED))
    body.append(text(px, T + 128, f"{run['final_loss']:.3f}", 22, INK, family=MONO))
    body.append(text(px, T + 176, "minimum", 13, MUTED))
    body.append(text(px, T + 204, f"{run['loss_min']:.3f}", 22, INK, family=MONO))
    save("loss", frame(W, H, "".join(body)))


# ---- flow matching loss and statistics ---------------------------------------

def flow():
    run = json.load(open(os.path.join(ROOT, "results/flow-matching/run.json")))
    raw = run["curve"]
    win = 50
    curve = [sum(raw[max(0, i - win + 1):i + 1]) / len(raw[max(0, i - win + 1):i + 1])
             for i in range(len(raw))]
    steps = run["steps"]
    body = []
    card_title(body, "Rectified flow loss",
               f"{steps} steps, 50-step moving average, 128 two-moon points, Euler 32")
    L, R, T, B = 90, 360, 130, 72
    W, H = 1200, 560
    pw, ph = W - L - R, H - T - B
    y0 = T + ph
    lo, hi = 0.6, 1.6

    def yv(v):
        import math
        t = (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo))
        return y0 - t * ph

    for tick in (0.6, 0.8, 1.0, 1.2, 1.4, 1.6):
        y = yv(tick)
        body.append(line(L, y, L + pw, y, GRID, 1))
        body.append(text(L - 14, y + 5, f"{tick:g}", 13, MUTED, anchor="end",
                         family=MONO))
    step = max(1, len(curve) // 600)
    pts = [f"{L + (i / (len(curve) - 1)) * pw:.1f},{yv(v):.1f}"
           for i, v in enumerate(curve) if i % step == 0 or i == len(curve) - 1]
    body.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{ORANGE}" '
                f'stroke-width="2.6" stroke-linejoin="round"/>')
    body.append(text(L + 8, yv(curve[0]) + 18, f"first {curve[0]:.3f}", 13, ORANGE,
                     family=MONO))
    body.append(text(L + pw / 2, y0 + 44, "training step", 13, MUTED,
                     anchor="middle"))

    px = L + pw + 60
    rows = (("last 100 steps", f"{run['last100']:.3f}"),
            ("sample mean", f"{run['sample_mean'][0]:+.3f}  {run['sample_mean'][1]:+.3f}"),
            ("sample std", f"{run['sample_std'][0]:.3f}  {run['sample_std'][1]:.3f}"),
            ("data mean", f"{run['data_mean'][0]:+.3f}  {run['data_mean'][1]:+.3f}"),
            ("data std", f"{run['data_std'][0]:.3f}  {run['data_std'][1]:.3f}"),
            ("inside the box", f"{run['inside_box']}/128"))
    y = T + 20
    for label, value in rows:
        body.append(text(px, y, label, 12, MUTED))
        body.append(text(px, y + 26, value, 19, INK, family=MONO))
        y += 68
    save("flow", frame(W, H, "".join(body)))


# ---- the DDIM samples --------------------------------------------------------

def samples():
    run = json.load(open(os.path.join(ROOT, "results/diffusion-vit/run.json")))
    px = run["per_sampler"]["ddim"]["samples"]
    body = []
    card_title(body, "Samples from the ViT denoiser",
               "64 DDIM samples, 8x8 pixels each, from the 16-step walk")
    tile, gap = 64, 8
    grid = 8 * tile + 7 * gap
    left = (1200 - grid) / 2
    top = 130

    def mix(c1, c2, t):
        a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
        b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
        return "#%02x%02x%02x" % tuple(int(a[j] + (b[j] - a[j]) * t) for j in range(3))

    for d in range(64):
        cx = left + (d % 8) * (tile + gap)
        cy = top + (d // 8) * (tile + gap)
        body.append(rect(cx, cy, tile, tile, PANEL, 4))
        for j in range(64):
            v = max(0.0, min(1.0, float(px[d * 64 + j])))
            body.append(rect(cx + (j % 8) * 8, cy + (j // 8) * 8, 8, 8,
                             mix("#f2eee7", "#5e5787", v)))
    body.append(text(600, top + grid + 34,
                     "each tile is one 8x8 digit, rendered from the samples in run.json",
                     12, MUTED, anchor="middle"))
    save("samples", frame(1200, 790, "".join(body)))


# ---- the plumbing split ------------------------------------------------------

def plumbing():
    before = [("build", 418, ORANGE), ("forward", 51, VIO), ("backward", 74, GREEN)]
    after = [("build", 7, ORANGE), ("forward", 53, VIO), ("backward", 71, GREEN)]
    body = []
    card_title(body, "One ViT training step, before and after the builder fix",
               "milliseconds, median of three runs; the values are bit-identical")
    W, H = 1200, 470
    L, M = 210, 160
    span = 900

    def bar(y, title, parts, total, maxv):
        body.append(text(L - 18, y + 34, title, 15, INK, anchor="end",
                         weight="600"))
        x = L
        for name, v, color in parts:
            w = v / maxv * span
            body.append(rect(x, y, w, 52, color, 3, 0.9))
            if w > 64:
                body.append(text(x + 10, y + 32, f"{v}", 15, "#fff", family=MONO,
                                 weight="600"))
            x += w
        body.append(text(x + 16, y + 33, f"{total} ms", 17, INK, family=MONO,
                         weight="600"))

    bar(M, "before", before, 543, 543)
    bar(M + 110, "after", after, 131, 543)

    lx = L
    for name, color in (("build", ORANGE), ("forward", VIO),
                        ("backward + optimizer", GREEN)):
        body.append(rect(lx, H - 58, 14, 14, color, 3))
        body.append(text(lx + 22, H - 46, name, 13, MUTED))
        lx += 240
    save("plumbing", frame(W, H, "".join(body)))


def gallery():
    names = ["legibility", "loss", "samples", "flow", "plumbing"]
    items = "".join(
        f'<figure><img src="{n}.svg" alt="{n}"><figcaption>{n}</figcaption></figure>'
        for n in names)
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<style>
body{{margin:0;padding:40px;background:{BG};font-family:{SANS};color:{INK}}}
h1{{font-size:22px;margin:0 0 24px}}
figure{{margin:0 0 32px}} img{{width:100%;max-width:1200px;display:block}}
figcaption{{font-size:12px;color:{MUTED};margin-top:6px}}
</style></head><body><h1>bend-lab figures</h1>{items}</body></html>"""
    with open(os.path.join(OUT, "index.html"), "w") as f:
        f.write(html)


def main():
    legibility()
    loss()
    flow()
    plumbing()
    samples()
    gallery()


if __name__ == "__main__":
    main()
