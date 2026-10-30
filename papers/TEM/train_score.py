"""Train the TEM backbone as a denoising score network on flowers 64x64.

  reference/.venv-lucidrains/bin/python papers/TEM/train_score.py --steps 2000
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import load
from score_small import ScoreTEM
from train import grid_image

OUT = os.path.join(HERE, "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--sigma", type=float, default=0.2)
    ap.add_argument("--sample-steps", type=int, default=120)
    ap.add_argument("--eval-samples", type=int, default=64)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--tag", default="tem64-score")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    os.makedirs(OUT, exist_ok=True)
    train_x, train_y = load("train")
    test_x, test_y = load("test")
    train_x = torch.from_numpy(train_x).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    test_x = torch.from_numpy(test_x).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    train_y = torch.from_numpy(train_y).long()
    test_y = torch.from_numpy(test_y).long()

    model = ScoreTEM()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    curve, t0 = [], time.time()
    for step in range(args.steps):
        idx = torch.randint(0, len(train_x), (args.batch,))
        x = train_x[idx]
        if torch.rand(()) < 0.5:
            x = torch.flip(x, dims=[3])
        y = train_y[idx]
        eps = torch.randn_like(x)
        xn = x + args.sigma * eps
        pred = model(xn, y)
        target = -eps / args.sigma
        loss = F.mse_loss(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        curve.append({"step": step, "loss": float(loss.detach())})
        if step % args.log_every == 0:
            print(f"{step:5d}  loss {float(loss):.4f}  "
                  f"{(time.time() - t0) / (step + 1):.2f}s/step", flush=True)

    torch.save({"model": model.state_dict()}, os.path.join(OUT, f"{args.tag}.pt"))
    model.eval()
    torch.manual_seed(1)
    sel = torch.randint(0, len(test_x), (args.eval_samples,))
    y = test_y[sel]
    x = torch.randn(args.eval_samples, 3, 64, 64)
    alpha = args.sigma ** 2
    for i in range(args.sample_steps):
        with torch.no_grad():
            s = model(x, y)
        x = (x + alpha * s + (2 * alpha) ** 0.5 * torch.randn_like(x)).clamp(-1, 1)
    grid_image(x, os.path.join(OUT, f"{args.tag}-samples.png"))
    np.save(os.path.join(OUT, f"{args.tag}-samples.npy"), x.numpy())

    real = test_x.reshape(len(test_x), -1)
    nn = torch.cdist(x.reshape(len(x), -1), real).min(1).values
    record = {"tag": args.tag, "params": sum(p.numel() for p in model.parameters()),
              "steps": args.steps, "sigma": args.sigma, "alpha": alpha,
              "sample_steps": args.sample_steps,
              "loss_first": curve[0]["loss"], "loss_last": curve[-1]["loss"],
              "loss_last50": float(np.mean([c["loss"] for c in curve[-50:]])),
              "sample_mean": x.mean((0, 2, 3)).tolist(),
              "sample_std": x.std((0, 2, 3)).tolist(),
              "nn_test_mean": float(nn.mean()), "seconds": time.time() - t0,
              "curve": curve[::max(1, len(curve) // 200)]}
    json.dump(record, open(os.path.join(OUT, f"{args.tag}.json"), "w"), indent=2)
    print(json.dumps({k: record[k] for k in ("loss_first", "loss_last50",
                                             "nn_test_mean", "seconds")}, indent=2))


if __name__ == "__main__":
    main()
