"""Writes a README and a plot for every example.

  reference/.venv/bin/python reference/make_example_readmes.py

Reads the latest report of each experiment task in results/latest/, runs the
two examples that are not tasks (logistic, attention) for their numbers, and
writes examples/<name>/README.md and examples/<name>/plot.svg.
"""
import json
import os
import re
import subprocess
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
if os.path.isdir(ENV_DIR):
    ENV["BEND_LIB"] = ENV_DIR


def latest(task):
    with open(os.path.join(ROOT, "results/latest", task + ".json")) as f:
        return json.load(f)


def metric(task, name):
    row = latest(task)["metrics"][name]
    return row.get("bend"), row.get("reference")


def esc(text):
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def bar_svg(path, title, labels, values, reference=None, unit=""):
    width, height = 640, 300
    left, right, top, bottom = 70, 20, 40, 60
    plot_w, plot_h = width - left - right, height - top - bottom
    top_value = max(list(values) + (list(reference) if reference else [])) * 1.15 or 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="24" font-family="sans-serif" font-size="14">{esc(title)}</text>',
    ]
    n = len(values)
    slot = plot_w / max(1, n)
    for i, (label, value) in enumerate(zip(labels, values)):
        x = left + slot * i + slot * 0.15
        w = slot * 0.7 if reference is None else slot * 0.32
        h = plot_h * value / top_value
        y = top + plot_h - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="#c0392b"/>')
        parts.append(f'<text x="{x + w/2:.1f}" y="{y - 5:.1f}" text-anchor="middle" font-family="sans-serif" '
                     f'font-size="11">{value:.4g}{esc(unit)}</text>')
        parts.append(f'<text x="{left + slot * i + slot/2:.1f}" y="{height - 24}" text-anchor="middle" '
                     f'font-family="sans-serif" font-size="11">{esc(label)}</text>')
        if reference is not None:
            rv = reference[i]
            rh = plot_h * rv / top_value
            ry = top + plot_h - rh
            parts.append(f'<rect x="{x + w + 4:.1f}" y="{ry:.1f}" width="{w:.1f}" height="{rh:.1f}" fill="#2c3e50"/>')
            parts.append(f'<text x="{x + w + 4 + w/2:.1f}" y="{ry - 5:.1f}" text-anchor="middle" '
                         f'font-family="sans-serif" font-size="11">{rv:.4g}{esc(unit)}</text>')
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{width-right}" y2="{top + plot_h}" stroke="#333"/>')
    if reference is not None:
        parts.append(f'<text x="{left}" y="{height - 6}" font-family="sans-serif" font-size="10" fill="#c0392b">Bend</text>')
        parts.append(f'<text x="{left + 50}" y="{height - 6}" font-family="sans-serif" font-size="10" fill="#2c3e50">reference</text>')
    parts.append("</svg>")
    open(path, "w").write("\n".join(parts))


def line_svg(path, title, series):
    width, height = 640, 300
    left, right, top, bottom = 60, 20, 40, 50
    plot_w, plot_h = width - left - right, height - top - bottom
    all_values = [v for _, values in series for v in values]
    lo, hi = min(all_values), max(all_values)
    span = hi - lo or 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="24" font-family="sans-serif" font-size="14">{esc(title)}</text>',
    ]
    colors = ["#c0392b", "#2c3e50"]
    for n, (name, values) in enumerate(series):
        points = " ".join(
            f"{left + plot_w * i / max(1, len(values) - 1):.1f},"
            f"{top + plot_h * (1 - (v - lo) / span):.1f}" for i, v in enumerate(values))
        parts.append(f'<polyline fill="none" stroke="{colors[n % 2]}" stroke-width="2" points="{points}"/>')
        parts.append(f'<text x="{left + 10 + n*110}" y="{top + 15}" font-family="sans-serif" font-size="12" '
                     f'fill="{colors[n % 2]}">{esc(name)}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>')
    parts.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>')
    parts.append(f'<text x="{width/2}" y="{height-8}" text-anchor="middle" font-family="sans-serif" font-size="11">step</text>')
    parts.append("</svg>")
    open(path, "w").write("\n".join(parts))


def attention_heatmap(path):
    binary = os.path.join(tempfile.mkdtemp(), "mlbend-attention-readme")
    build = subprocess.run([BEND, "-o", binary, os.path.join(ROOT, "examples/attention/main.bend")],
                           capture_output=True, text=True, env=ENV, cwd=ROOT)
    if build.returncode != 0:
        return None
    out = subprocess.run([binary], capture_output=True, text=True, env=ENV, cwd=ROOT)
    rows = [[float(v) for v in m.group(1).split()]
            for m in re.finditer(r"^row \d+: (.*)$", out.stdout, re.M)]
    if not rows:
        return None
    width, height = 640, 300
    left, top, cell_w, cell_h = 80, 40, 55, 45
    lo = min(min(r) for r in rows)
    hi = max(max(r) for r in rows)
    span = hi - lo or 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="24" font-family="sans-serif" font-size="14">'
        'Attention output (four tokens, eight features)</text>',
    ]
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            t = (value - lo) / span
            r, g, b = int(250 - 170 * t), int(245 - 120 * t), int(240 - 60 * t)
            parts.append(f'<rect x="{left + j*cell_w}" y="{top + i*cell_h}" width="{cell_w-2}" '
                         f'height="{cell_h-2}" fill="rgb({r},{g},{b})"/>')
            parts.append(f'<text x="{left + j*cell_w + (cell_w-2)/2}" y="{top + i*cell_h + 25}" '
                         f'text-anchor="middle" font-family="sans-serif" font-size="9">{value:.3f}</text>')
        parts.append(f'<text x="{left-8}" y="{top + i*cell_h + 25}" text-anchor="end" '
                     f'font-family="sans-serif" font-size="11">token {i}</text>')
    parts.append("</svg>")
    open(path, "w").write("\n".join(parts))
    return True


def write(name, title, body, plot=True):
    path = os.path.join(ROOT, "examples", name, "README.md")
    image = "![plot](plot.svg)\n\n" if plot else ""
    open(path, "w").write(f"# {title}\n\n{body}\n\n{image}".rstrip() + "\n")


def main():
    # task-backed examples
    acc, ref = metric("cancer-knn", "accuracy")
    bar_svg(os.path.join(ROOT, "examples/knn/plot.svg"), "Accuracy on breast cancer",
            ["Bend", "scikit-learn"], [acc, ref])
    write("knn", "k-nearest neighbours",
          "Three nearest neighbours over the breast cancer dataset, standardised, "
          "with the first four fifths as the training rows. No gradients and no library: "
          "a distance and a vote.\n\n"
          "```console\n$ bendenv run -- bend examples/knn/main.bend\n```\n\n"
          f"The accuracy is **{acc:.6f}**, against scikit-learn's **{ref:.6f}** "
          "on the same split (`reference/test_knn.py`).")

    acc, ref = metric("cancer-naive-bayes", "accuracy")
    bar_svg(os.path.join(ROOT, "examples/naive_bayes/plot.svg"), "Accuracy on breast cancer",
            ["Bend", "scikit-learn"], [acc, ref])
    write("naive_bayes", "Gaussian naive Bayes",
          "A Gaussian per feature per class, trained on the first four fifths of the "
          "breast cancer dataset. The gradient-free baseline, with no standardising "
          "(a Gaussian is affine-invariant).\n\n"
          "```console\n$ bendenv run -- bend examples/naive_bayes/main.bend\n```\n\n"
          f"The accuracy is **{acc:.6f}**, the same as scikit-learn's `GaussianNB` "
          f"(**{ref:.6f}**).")

    acc, ref = metric("iris-crossval", "mean_accuracy")
    bar_svg(os.path.join(ROOT, "examples/crossval/plot.svg"), "Mean accuracy over five folds",
            ["Bend", "scikit-learn"], [acc, ref])
    write("crossval", "Cross-validation, without leakage",
          "The whole chain: iris from HuggingFace, the scaler fitted by "
          "`mlbend-pipeline` on the training rows of each fold, logistic regression "
          "with bend-ml's autodiff, the folds from `mlbend-select`, the score from "
          "`mlbend-metrics`.\n\n"
          "```console\n$ bendenv run -- bend examples/crossval/main.bend\n```\n\n"
          f"Mean accuracy **{acc:.4f}**, against scikit-learn's `Pipeline(StandardScaler, "
          f"LogisticRegression)` at **{ref:.4f}** on the same folds: the program prints "
          "the fold of every row and the reference runs those folds, so the two numbers "
          "agree to 5.9e-8 instead of agreeing only about the procedure.")

    acc, ref = metric("iris-mlp", "accuracy")
    bar_svg(os.path.join(ROOT, "examples/mlp/plot.svg"), "Accuracy on iris",
            ["Bend", "scikit-learn"], [acc, ref])
    write("mlp", "A two-layer network",
          "Eight rectified hidden units over iris with bend-ml's tensors and autodiff: "
          "`AG.linear`, `AG.mask.rows`, `AG.linear` again, cross-entropy, and the "
          "backward chain through both layers. 400 SGD steps.\n\n"
          "```console\n$ bendenv run -- bend examples/mlp/main.bend\n```\n\n"
          f"Accuracy **{acc:.6f}**, the same as scikit-learn's `MLPClassifier(8)` "
          f"(**{ref:.6f}**). The loss falls from 1.1172 to 0.1107.")

    inertia, _ = metric("wine-kmeans", "inertia")
    sizes = latest("wine-kmeans")["metrics"]["cluster_sizes"]["bend"]
    bar_svg(os.path.join(ROOT, "examples/kmeans/plot.svg"), "Rows per cluster (wine)",
            [f"cluster {i}" for i in range(len(sizes))], sizes)
    write("kmeans", "k-means over wine",
          "Three clusters, ten rounds, through `mlbend-cluster` after standardising "
          "the columns. The algorithm lives in the package; the example holds the data "
          "and the standardising.\n\n"
          "```console\n$ bendenv run -- bend examples/kmeans/main.bend\n```\n\n"
          f"Inertia **{inertia:.2f}**; cluster sizes **{' '.join(str(s) for s in sizes)}**. "
          "The same numbers as the NumPy implementation (`reference/test_kmeans.py`).")

    ratio, ref = metric("wine-pca", "explained_variance_ratio")
    bar_svg(os.path.join(ROOT, "examples/pca/plot.svg"), "First component: explained variance",
            ["Bend", "scikit-learn"], [ratio, ref])
    write("pca", "The first principal component",
          "The covariance matrix of the standardised wine features and a hundred power "
          "iterations for its first eigenvector. A transformer, not a trainer.\n\n"
          "```console\n$ bendenv run -- bend examples/pca/main.bend\n```\n\n"
          f"Explained variance ratio **{ratio:.6f}**, against scikit-learn's "
          f"**{ref:.6f}**.")

    acc, ref = metric("heart-naive-bayes", "accuracy")
    bar_svg(os.path.join(ROOT, "examples/naive_bayes_heart/plot.svg"), "Accuracy on heart failure",
            ["Bend", "scikit-learn"], [acc, ref])
    write("naive_bayes_heart", "Gaussian naive Bayes on heart failure",
          "The same baseline as `naive_bayes`, on a second dataset: 299 rows of twelve "
          "features, the label `is_dead`, the first four fifths for training.\n\n"
          "```console\n$ bendenv run -- bend examples/naive_bayes_heart/main.bend\n```\n\n"
          f"Accuracy **{acc:.6f}**, against scikit-learn's **{ref:.6f}** on the same split.")

    # logistic: a line of its four losses
    line_svg(os.path.join(ROOT, "examples/logistic/plot.svg"), "Loss over four SGD steps",
             [("logistic", [1.0986128, 1.086162, 1.0759476, 1.0671557])])
    write("logistic", "Logistic regression",
          "A logistic regression over iris with bend-ml's tensors and autodiff: "
          "`AG.linear`, `AG.ce_loss`, `AG.ce_grad`, `AG.linear_bwd`, `AG.sgd`. "
          "The learning rate is 0.01 because the features are not standardised.\n\n"
          "```console\n$ bendenv run -- bend examples/logistic/main.bend\n```\n\n"
          "Loss **1.0986128 → 1.086162 → 1.0759476 → 1.0671557** over four steps.")

    # attention: heatmap of its output
    attention_heatmap(os.path.join(ROOT, "examples/attention/plot.svg"))
    write("attention", "Single-head self-attention",
          "The core of a transformer in plain Bend: Q, K and V projections, the scaled "
          "dot-product scores, a row softmax and the weighted sum of V. Four tokens of "
          "eight features, fixed weights; every output agrees with the NumPy reference "
          "to 1.5e-8.\n\n"
          "```console\n$ bendenv run -- bend examples/attention/main.bend\n```\n\n"
          "The heatmap is the 4 × 8 output; `reference/test_attention.py` checks every "
          "number and the checksum.")

    # transformer: use the saved loss graph
    src = os.path.join(ROOT, "results/transformer/loss.svg")
    dst = os.path.join(ROOT, "examples/transformer/plot.svg")
    if os.path.exists(src):
        open(dst, "w").write(open(src).read())
    bend = latest("transformer-periodic")["metrics"]
    first = bend["first_loss"]["bend"]
    final = bend["final_loss"]["bend"]
    accuracy = bend["accuracy"]["bend"]
    write("transformer", "A trained transformer",
          "A one-block causal transformer on `bendtorch@1.0.1.0`: embedding, RMSNorm, "
          "single-head causal attention, output projection, RMSNorm, linear head. "
          "AdamW, 200 steps, next-token prediction on the periodic sequence 0..7.\n\n"
          "```console\n$ bendenv run -- bend examples/transformer/main.bend\n```\n\n"
          f"Loss **{first:.4f} → {final:.4f}**, next-token accuracy **{accuracy:.4f}**. "
          "The NumPy twin starts at the same loss and ends at 0.1351 with accuracy "
          "0.9375; the full curve and the manifest are in "
          "[`results/transformer/`](../../results/transformer/).")
    # the two diffusion examples: their saved picture beside their report's numbers
    src = os.path.join(ROOT, "results/diffusion/scatter.svg")
    if os.path.exists(src):
        open(os.path.join(ROOT, "examples/diffusion/plot.svg"), "w").write(open(src).read())
    bend = latest("diffusion-2d")["metrics"]
    first = bend["first_loss"]["bend"]
    final = bend["final_loss"]["bend"]
    write("diffusion", "A DDPM denoiser on two moons",
          "A three-linear-layer SiLU denoiser on `bendtorch@1.0.1.0`, trained by AdamW for "
          "4000 steps to predict the noise of a 64-step linear-β forward process over two "
          "moons, then sampled two ways: the ancestral loop over all 64 rungs, and DDIM "
          "over every 4th rung — 16 model calls instead of 64, and at η = 0 a step reads "
          "no noise at all. Every random number is a pure function of a counter "
          "(murmur3's fmix32 over the step salt and the index, Box-Muller over 24-bit "
          "uniforms), so the NumPy twin draws the same ones.\n\n"
          "```console\n$ bendenv run -- bend examples/diffusion/main.bend\n```\n\n"
          f"First loss **{first:.4f}** against the twin's — the gate — and final loss "
          f"**{final:.4f}**, reported. Both clouds' statistics are report-mode: the "
          "optimizers round differently and a 64-step loop amplifies a ulp. The curve, "
          "the cloud and the manifest are in [`results/diffusion/`](../../results/diffusion/).")

    src = os.path.join(ROOT, "results/diffusion-digits/grid.svg")
    if os.path.exists(src):
        open(os.path.join(ROOT, "examples/diffusion_digits/plot.svg"), "w").write(open(src).read())
    bend = latest("diffusion-digits")["metrics"]
    first = bend["first_loss"]["bend"]
    write("diffusion_digits", "A DDPM denoiser on digits",
          "The same denoiser over real data: the first 1024 rows of `mstz/optdigits` "
          "(pinned revision, its SHA-256 in the manifest), 8x8 digits scaled to [0, 1]. "
          "64 pixels in, a 64-wide SiLU body, 64-step schedule, 3000 full-batch AdamW "
          "steps, then 64 fresh digits by the ancestral loop and DDIM over every 4th rung. "
          "The image shows real digits on the left and the samples on the right, and the "
          "samplers are checked element by element against the NumPy twin by "
          "`reference/test_diffusion_digits.py` (a frozen-weights probe, because mean and "
          "std over a grid cannot see a reordered array).\n\n"
          "```console\n$ bendenv run -- bend examples/diffusion_digits/main.bend\n```\n\n"
          f"First loss **{first:.2f}** against the NumPy twin's — the gate — and the data "
          "mean to 4.6e-9. The final loss, both clouds' statistics and the grid are "
          "reported, not gated: the underfit denoiser makes the sampler expand "
          "geometrically, which is the known follow-up (a residual head), and the "
          "variance is factored by its largest deviation so the statistics stay finite "
          "at that magnitude. The curve, the grid and the manifest are in "
          "[`results/diffusion-digits/`](../../results/diffusion-digits/).")
    print("wrote example READMEs and plots")


if __name__ == "__main__":
    main()
