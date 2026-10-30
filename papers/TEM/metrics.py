"""FID, KID, precision and recall for a TEM sample set, at 64x64.

  reference/.venv-lucidrains/bin/python papers/TEM/metrics.py \
      papers/TEM/results/tem64-samples.npy

Inception runs on 64x64 images upscaled to its 299 input. This is the same
metric family as the paper, not the paper's measurement: the paper's FID is
ImageNet 256 against 50K real images.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch_fidelity

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import load


def to_uint8(x):
    x = np.clip((x + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return torch.from_numpy(x).permute(0, 3, 1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples")
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    smp = np.load(args.samples)
    real, _ = load("test")
    fake = to_uint8(smp)
    real_t = torch.from_numpy(real).permute(0, 3, 1, 2)
    out = torch_fidelity.calculate_metrics(
        input1=fake, input2=real_t, cuda=False, batch_size=args.batch,
        fid=True, kid=True, isc=True, prc=True, verbose=False)
    record = {
        "samples": os.path.basename(args.samples),
        "n_fake": int(len(smp)), "n_real": int(len(real)),
        "fid": float(out["frechet_inception_distance"]),
        "kid": float(out["kernel_inception_distance_mean"]),
        "inception_score": float(out["inception_score_mean"]),
        "precision": float(out["precision"]),
        "recall": float(out["recall"]),
    }
    path = args.out or os.path.join(HERE, "results", "metrics.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
