"""Model selection and a classification report for the kNN task.

  reference/.venv/bin/python reference/select.py

The experiment layer's next layer: choose a parameter by cross-validation and
report the result. This is the Python side of the loop, on the same breast
cancer split the Bend kNN example uses; the Bend counterpart is the fixed
`k = 3` run in `examples/knn`.

Writes results/selection/knn.json, results/selection/knn.svg and
results/report/knn.md.
"""
import json
import os
import sys

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))
from test_knn import load  # noqa: E402


def bar_svg(path, title, labels, values, best):
    width, height = 640, 300
    left, right, top, bottom = 70, 20, 40, 60
    plot_w, plot_h = width - left - right, height - top - bottom
    top_value = max(values) * 1.1 or 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="24" font-family="sans-serif" font-size="14">{title}</text>',
    ]
    slot = plot_w / len(values)
    for i, (label, value) in enumerate(zip(labels, values)):
        h = plot_h * value / top_value
        x = left + slot * i + slot * 0.15
        y = top + plot_h - h
        color = "#c0392b" if label == best else "#2c3e50"
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{slot*0.7:.1f}" height="{h:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{x + slot*0.35:.1f}" y="{y - 5:.1f}" text-anchor="middle" '
                     f'font-family="sans-serif" font-size="10">{value:.3f}</text>')
        parts.append(f'<text x="{x + slot*0.35:.1f}" y="{height - 24}" text-anchor="middle" '
                     f'font-family="sans-serif" font-size="10">{label}</text>')
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{width-right}" y2="{top + plot_h}" stroke="#333"/>')
    parts.append(f'<text x="{left}" y="{height - 6}" font-family="sans-serif" font-size="10" fill="#c0392b">'
                 f'chosen: {best}</text>')
    parts.append("</svg>")
    open(path, "w").write("\n".join(parts))


def stratified_folds(y, k=5, seed=42):
    """Round-robin folds within each class, no scipy."""
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        for i, row in enumerate(idx):
            folds[i % k].append(int(row))
    return [np.array(sorted(f), dtype=int) for f in folds]


def report_of(true, pred):
    tp = int(((pred == 1) & (true == 1)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    tn = int(((pred == 0) & (true == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    m_precision = tn / (tn + fn) if tn + fn else 0.0
    m_recall = tn / (tn + fp) if tn + fp else 0.0
    m_f1 = 2 * m_precision * m_recall / (m_precision + m_recall) if m_precision + m_recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "m_precision": m_precision, "m_recall": m_recall, "m_f1": m_f1,
            "accuracy": (tp + tn) / max(1, len(true))}


def main():
    x, y = load()
    ks = list(range(1, 16))
    folds = stratified_folds(y)
    scores = []
    for k in ks:
        accs = []
        for held in folds:
            train = np.setdiff1d(np.arange(len(y)), held)
            scaler = StandardScaler().fit(x[train])
            model = KNeighborsClassifier(n_neighbors=k).fit(scaler.transform(x[train]), y[train])
            accs.append(float((model.predict(scaler.transform(x[held])) == y[held]).mean()))
        scores.append(float(np.mean(accs)))
    best_k = ks[int(np.argmax(scores))]

    split = len(x) * 4 // 5
    scaler = StandardScaler().fit(x[:split])
    model = KNeighborsClassifier(n_neighbors=best_k).fit(scaler.transform(x[:split]), y[:split])
    pred = model.predict(scaler.transform(x[split:]))
    report = report_of(y[split:], pred)

    sel_dir = os.path.join(ROOT, "results/selection")
    rep_dir = os.path.join(ROOT, "results/report")
    os.makedirs(sel_dir, exist_ok=True)
    os.makedirs(rep_dir, exist_ok=True)
    json.dump({"parameter": "k", "values": ks, "mean_accuracy": scores, "best_k": best_k,
               "best_accuracy": max(scores), "split": {"train": split, "test": len(x) - split}},
              open(os.path.join(sel_dir, "knn.json"), "w"), indent=2)
    bar_svg(os.path.join(sel_dir, "knn.svg"), "k-nearest neighbours: 5-fold accuracy",
            [str(k) for k in ks], scores, str(best_k))

    lines = ["# kNN classification report", "",
             f"Chosen k: **{best_k}** (cross-validated accuracy {max(scores):.4f})", "",
             "| class | precision | recall | f1 | support |", "|---|---:|---:|---:|---:|",
             f"| malignant | {report['m_precision']:.4f} | {report['m_recall']:.4f} | "
             f"{report['m_f1']:.4f} | {report['fp'] + report['tn']} |",
             f"| benign | {report['precision']:.4f} | {report['recall']:.4f} | {report['f1']:.4f} | "
             f"{report['tp'] + report['fn']} |",
             "", f"Accuracy: **{report['accuracy']:.4f}**", "",
             "Confusion matrix (rows true, columns predicted):", "",
             "| | malignant | benign |", "|---|---:|---:|",
             f"| malignant | {report['tn']} | {report['fp']} |",
             f"| benign | {report['fn']} | {report['tp']} |", ""]
    open(os.path.join(rep_dir, "knn.md"), "w").write("\n".join(lines))
    print(f"best k {best_k}, cv accuracy {max(scores):.4f}, test accuracy {report['accuracy']:.4f}")
    print(f"wrote {sel_dir}/knn.json, knn.svg and {rep_dir}/knn.md")


if __name__ == "__main__":
    main()
