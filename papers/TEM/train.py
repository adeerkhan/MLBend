"""Train the small pixel-space TEM on Oxford Flowers 64x64.

  reference/.venv-lucidrains/bin/python papers/TEM/train.py --steps 2000

This is the Python arm of the paper comparison. It follows the paper's
contrastive-divergence loss and Langevin ordering on 64x64 pixels, not on
SD-VAE latents, so its numbers are not the paper's numbers.
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
from tem_small import TEM, count_params, langevin

OUT = os.path.join(HERE, "results")


def to_tensor(images):
    x = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    return x


def grid_image(samples, path, n=8):
    from PIL import Image
    x = ((samples.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).numpy()
    tile = 64
    out = np.zeros((n * tile, n * tile, 3), dtype=np.uint8)
    for i in range(min(n * n, len(x))):
        out[(i // n) * tile:(i // n + 1) * tile, (i % n) * tile:(i % n + 1) * tile] = x[i]
    Image.fromarray(out).save(path)


def sample(model, labels, init, steps, step_size, noise_std):
    x = init.clone()
    x = langevin(model, x, labels, steps - 1, step_size, noise_std)
    with torch.enable_grad():
        x = x.detach().requires_grad_(True)
        e = model.energy(x, labels).sum()
        g = torch.autograd.grad(e, x)[0]
    return (x - step_size * g).detach().clamp(-1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--step-size", type=float, default=0.01)
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--sample-steps", type=int, default=40)
    ap.add_argument("--buffer", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--eval-samples", type=int, default=256)
    ap.add_argument("--tag", default="tem64")
    ap.add_argument("--pra-coeff", type=float, default=0.5)
    ap.add_argument("--pra-layer", type=int, default=1)
    ap.add_argument("--dino", default="dinov2_vits14")
    ap.add_argument("--var-coeff", type=float, default=0.1)
    ap.add_argument("--dsm-coeff", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--buffer-init", default="data")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(OUT, exist_ok=True)

    train_x, train_y = load("train")
    test_x, test_y = load("test")
    train_x = to_tensor(train_x)
    test_x = to_tensor(test_x)
    labels_all = torch.from_numpy(train_y).long()
    test_labels = torch.from_numpy(test_y).long()

    model = TEM(depth=args.depth, dim=args.dim, heads=args.heads,
                pra_layer=args.pra_layer)
    dino = torch.hub.load("facebookresearch/dinov2", args.dino)
    dino.eval()
    for p in dino.parameters():
        p.requires_grad_(False)
    dino_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    dino_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    if args.buffer_init == "data":
        bi = torch.randint(0, len(train_x), (args.buffer,))
        buffer = (train_x[bi] + 0.1 * torch.randn(args.buffer, 3, 64, 64)).clamp(-1, 1)
    else:
        buffer = torch.randn(args.buffer, 3, 64, 64).clamp(-1, 1)
    meta = {"tag": args.tag, "params": count_params(model), "steps": args.steps,
            "batch": args.batch, "lr": args.lr, "step_size": args.step_size,
            "noise": args.noise, "sample_steps": args.sample_steps,
            "depth": args.depth, "dim": args.dim, "heads": args.heads,
            "data": "oxford-flowers-64", "train": int(len(train_x)), "test": int(len(test_x))}
    curve, t0 = [], time.time()

    for step in range(args.steps):
        idx = torch.randint(0, len(train_x), (args.batch,))
        pos = train_x[idx].clone()
        if torch.rand(()) < 0.5:
            pos = torch.flip(pos, dims=[3])
        y = labels_all[idx]

        bidx = torch.randint(0, args.buffer, (args.batch,))
        neg = buffer[bidx].clone()
        neg = langevin(model, neg, y, args.sample_steps - 1,
                       args.step_size, args.noise)
        neg = neg.detach().requires_grad_(True)
        with torch.enable_grad():
            e = model.energy(neg, y).sum()
            g = torch.autograd.grad(e, neg)[0]
        neg = (neg - args.step_size * g).detach().clamp(-1, 1)
        model.train()

        pos_out, pra = model(pos, y, pra_layer=args.pra_layer)
        neg_out = model(neg, y)
        pos_e = pos_out.sum(1)
        neg_e = neg_out.sum(1)
        cd_loss = ((pos_out ** 2).sum(1) + (neg_out ** 2).sum(1) + pos_e - neg_e).mean()
        with torch.no_grad():
            dino_in = F.interpolate((pos + 1) / 2, size=(112, 112), mode="bicubic",
                                    align_corners=False)
            dino_in = (dino_in - dino_mean) / dino_std
            feat = dino.forward_features(dino_in)["x_norm_patchtokens"]
            feat = feat - 0.6 * feat.mean(1, keepdim=True)
            feat = feat / (feat.std(1, keepdim=True) + 1e-6)
        pra_loss = 1 - F.cosine_similarity(feat, pra, dim=-1).mean()
        var_term = pos_e.std() + neg_e.std()
        eps = torch.randn_like(pos)
        noisy = (pos + args.sigma * eps).detach().requires_grad_(True)
        with torch.enable_grad():
            e_n = model.energy(noisy, y).sum()
            g_n = torch.autograd.grad(e_n, noisy, create_graph=True)[0]
        dsm = ((g_n - eps / args.sigma) ** 2).mean()
        loss = (cd_loss + args.pra_coeff * pra_loss - args.var_coeff * var_term
                + args.dsm_coeff * dsm)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        with torch.no_grad():
            buffer[bidx] = neg
        curve.append({"step": step, "loss": float(loss.detach()),
                      "cd": float(cd_loss.detach()), "pra": float(pra_loss.detach()),
                      "var": float(var_term.detach()), "dsm": float(dsm.detach()),
                      "pos": float(pos_e.mean().detach()),
                      "neg": float(neg_e.mean().detach())})
        if step % args.log_every == 0:
            print(f"{step:5d}  loss {float(loss): .4f}  cd {float(cd_loss): .4f}  "
                  f"pra {float(pra_loss): .3f}  pos {float(pos_e.mean()): .3f}  "
                  f"neg {float(neg_e.mean()): .3f}  "
                  f"{(time.time() - t0) / (step + 1):.2f}s/step", flush=True)

    model.eval()
    torch.manual_seed(args.seed + 1)
    sel = torch.randint(0, len(test_x), (args.eval_samples,))
    init = torch.randn(args.eval_samples, 3, 64, 64).clamp(-1, 1)
    y = test_labels[sel]
    smp = sample(model, y, init, args.sample_steps, args.step_size, args.noise)
    grid_image(smp, os.path.join(OUT, f"{args.tag}-samples.png"))
    np.save(os.path.join(OUT, f"{args.tag}-samples.npy"), smp.numpy())
    torch.save({"model": model.state_dict(), "labels": y, "init": init},
               os.path.join(OUT, f"{args.tag}.pt"))

    flat = smp.reshape(len(smp), -1)
    real = test_x.reshape(len(test_x), -1)
    nn = torch.cdist(flat, real).min(1).values
    record = dict(meta)
    record.update({
        "seconds": time.time() - t0,
        "loss_first": curve[0]["loss"],
        "loss_final": curve[-1]["loss"],
        "loss_last50": float(np.mean([c["loss"] for c in curve[-50:]])),
        "cd_last50": float(np.mean([c["cd"] for c in curve[-50:]])),
        "pra_last50": float(np.mean([c["pra"] for c in curve[-50:]]))
        if "pra" in curve[-1] else None,
        "pos_energy": curve[-1]["pos"],
        "neg_energy": curve[-1]["neg"],
        "sample_mean": smp.mean((0, 2, 3)).tolist(),
        "sample_std": smp.std((0, 2, 3)).tolist(),
        "nn_test_mean": float(nn.mean()),
        "curve": curve[::max(1, len(curve) // 200)],
    })
    with open(os.path.join(OUT, f"{args.tag}.json"), "w") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    print(json.dumps({k: record[k] for k in ("params", "loss_first", "loss_final",
                                             "loss_last50", "pos_energy", "neg_energy",
                                             "nn_test_mean", "seconds")}, indent=2))
    print("samples:", os.path.join(OUT, f"{args.tag}-samples.png"))


if __name__ == "__main__":
    main()
