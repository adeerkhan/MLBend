"""The k-means over the wine dataset, in NumPy, for the Bend run to check.

  reference/.venv/bin/python reference/kmeans_wine.py

The Bend program does the same thing: it fetches the wine dataset from
HuggingFace, standardises every column, starts from the first three rows
and runs ten rounds of Lloyd's algorithm. This script does the same on
the same data, so the inertia and the cluster sizes can be compared.
"""
import io
import os
import sys
import urllib.request

import numpy as np

URL = "https://huggingface.co/datasets/mstz/wine/resolve/main/wine/train.csv"


def load():
    """The wine dataset, as a matrix and a label column."""
    text = urllib.request.urlopen(URL, timeout=120).read().decode()
    lines = [l for l in text.split("\n") if l.strip()]
    rows = []
    for line in lines[1:]:
        cells = line.split(",")
        rows.append([float(c) for c in cells[:-1]])
    return np.array(rows, dtype=np.float64)


def standardise(x):
    """Every column centred and scaled by its own standard deviation."""
    return (x - x.mean(axis=0)) / x.std(axis=0)


def kmeans(x, k, rounds):
    """Lloyd's algorithm, starting from the first k rows."""
    centres = x[:k].copy()
    for _ in range(rounds):
        d = ((x[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        labels = d.argmin(axis=1)
        new = np.array([x[labels == c].mean(axis=0) for c in range(k)])
        centres = new
    d = ((x[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
    labels = d.argmin(axis=1)
    inertia = float(d[np.arange(len(x)), labels].sum())
    sizes = [int((labels == c).sum()) for c in range(k)]
    return inertia, sizes


def main():
    x = load()
    inertia, sizes = kmeans(standardise(x), 3, 10)
    print(f"rows: {len(x)}")
    print(f"inertia: {inertia}")
    print("cluster sizes:", " ".join(str(s) for s in sizes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
