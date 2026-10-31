#!/usr/bin/env python3
"""What a 64x64 flowers-flow run produced, from its stdout.

  reference/.venv/bin/python reference/flowers_report.py run.txt [--png out.png]

`examples/diffusion/flowers_flow/main.bend` prints `params: <n>`, a `loss:` line
every 50 steps, `final_loss: <l>`, then `samples:` and 512 rows of 64 numbers
(8 images, raster order: image i is rows i*64 .. i*64+63), then `samples_end`.

The script prints what the result has to be judged on:

  loss       first, last and minimum of the logged line
  samples    mean, std, min, max, and the fraction inside the training box
  data       the same statistics over the pinned 512 flowers, for comparison
  png        the eight images as one grid, 4 x 2, each 4x nearest-neighbour

The data statistics are read from the pinned export, and its sha256 is checked,
so a number here is a number about the data that produced the run.
"""
import hashlib
import os
import re
import struct
import sys
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.expanduser("~/.cache/bend-lab/flowers/flowers64-512-gray.u8")
DATA_SHA = {
    64: "6b9c4c66ad5c03939875557d8a77af02845bf924849b4964bcd46b69300a4956",
    256: "c58eab331d8dddf6b45426c95654755e672020631e18349c03e9c8909b25220e",
}
EDGE = 64          # overwritten from the sample block
IMAGES = 8


def png(path, arr):
    """A grayscale 8-bit PNG without an imaging library."""
    h, w = arr.shape
    raw = b"".join(b"\x00" + arr[i].tobytes() for i in range(h))

    def chunk(tag, data):
        body = struct.pack(">I", len(data)) + tag + data
        return body + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    out = b"\x89PNG\r\n\x1a\n"
    out += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
    out += chunk(b"IDAT", zlib.compress(raw, 9))
    out += chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(out)


def data_stats(size=64, holdout=0):
    path = os.path.expanduser(f"~/.cache/bend-lab/flowers/flowers{size}-512-gray.u8")
    if not os.path.exists(path):
        return None
    blob = open(path, "rb").read()
    got = hashlib.sha256(blob).hexdigest()
    x = np.frombuffer(blob, dtype=np.uint8).astype(np.float32)
    x = x / np.float32(127.5) - np.float32(1.0)
    return got, x, holdout


def legibility(samples, train):
    """Each sample's best absolute correlation with any training image.

    A legibility proxy in the shape the digits arm uses: random images score a
    low constant, a blurry but real image scores clearly above it.
    """
    s = samples - samples.mean(1, keepdims=True)
    t = train - train.mean(1, keepdims=True)
    s = s / (np.linalg.norm(s, axis=1, keepdims=True) + 1e-12)
    t = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-12)
    return np.abs(s @ t.T).max(1)


def grid(images, scale=4):
    """8 images of EDGE x EDGE into 4 x 2 tiles, nearest neighbour."""
    tiles = []
    for img in images:
        up = np.repeat(np.repeat(img, scale, 0), scale, 1)
        tiles.append(up)
    if len(tiles) == 1:
        return tiles[0]
    half = (len(tiles) + 1) // 2
    top = np.concatenate(tiles[:half], axis=1)
    if len(tiles) == half:
        return top
    bottom = np.concatenate(tiles[half:] + [np.zeros_like(tiles[0])] * (half - len(tiles[half:])), axis=1)
    return np.concatenate([top, bottom], axis=0)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    text = open(sys.argv[1]).read()

    losses = [float(v) for v in re.findall(r"^loss: ([0-9.eE+-]+)$", text, re.M)]
    final = re.search(r"^final_loss: ([0-9.eE+-]+)$", text, re.M)
    params = re.search(r"^params: (\d+)$", text, re.M)
    block = re.search(r"^samples:\n(.*?)^samples_end$", text, re.M | re.S)

    print(f"params: {params.group(1) if params else 'not printed'}")
    print(f"loss lines: {len(losses)}")
    if losses:
        print(f"  first {losses[0]:.4f}  last {losses[-1]:.4f}  min {min(losses):.4f}")
    if final:
        print(f"  final {float(final.group(1)):.4f}")
    if not block:
        raise SystemExit("no sample block: did the run reach the sampler?")

    rows = [r for r in block.group(1).splitlines() if r.strip()]
    v = np.array([[float(x) for x in r.split()] for r in rows], dtype=np.float32)
    # the block is one row of pixels per token: columns are EDGE, and 64 rows
    # make one image. Both are read from the run, so a preset change needs no
    # edit here.
    if v.shape[1] not in (32, 64):
        raise SystemExit(f"{v.shape[1]} columns, expected 32 or 64")
    global EDGE, IMAGES
    EDGE = v.shape[1]
    if len(rows) % EDGE:
        raise SystemExit(f"{len(rows)} rows is not a whole number of {EDGE}-row images")
    IMAGES = len(rows) // EDGE

    images = v.reshape(IMAGES, EDGE, EDGE)
    print(f"samples: n {v.size}  mean {v.mean():+.4f}  std {v.std():.4f}  "
          f"min {v.min():+.4f}  max {v.max():+.4f}  in [-1,1]: "
          f"{int(((v >= -1) & (v <= 1)).sum())}/{v.size}")

    holdout = 0
    if "--holdout" in sys.argv:
        holdout = int(sys.argv[sys.argv.index("--holdout") + 1])
    st = data_stats(EDGE, holdout)
    if st is None:
        print(f"data: {DATA} not found, no comparison")
        sha_ok = False
    else:
        got, x, holdout = st
        sha_ok = got == DATA_SHA.get(EDGE)
        print(f"data: mean {x.mean():+.4f}  std {x.std():.4f}  "
              f"min {x.min():+.4f}  max {x.max():+.4f}  sha256 {got[:16]}… "
              f"{'pinned' if sha_ok else 'NOT the pinned export'}")
        print(f"  per-image std {np.round(images.reshape(IMAGES, -1).std(1), 3).tolist()}")
        flat = images.reshape(IMAGES, -1)
        allx = x.reshape(-1, EDGE * EDGE)
        if holdout:
            # the last holdout images were never trained on: scoring against them
            # measures the model, scoring against the trained part measures its
            # memory, and the gap between the two is the honest number
            print(f"held out: the last {holdout} images; trained on the first {len(allx) - holdout}")
            lt = legibility(flat, allx[:len(allx) - holdout])
            lh = legibility(flat, allx[len(allx) - holdout:])
            print(f"legibility: trained {lt.mean():.3f} (min {lt.min():.3f})  "
                  f"held out {lh.mean():.3f} (min {lh.min():.3f})  gap {lt.mean() - lh.mean():+.3f}")
            rng = np.random.default_rng(0)
            base = legibility(rng.normal(size=(IMAGES, EDGE * EDGE)).astype(np.float32),
                              allx[len(allx) - holdout:])
            print(f"  random baseline on the held-out set {base.mean():.3f}")
            out = "results/flowers/samples.png"
            for i, a in enumerate(sys.argv):
                if a == "--png" and i + 1 < len(sys.argv):
                    out = sys.argv[i + 1]
            os.makedirs(os.path.dirname(os.path.join(ROOT, out)) or ".", exist_ok=True)
            px = np.clip((images + 1.0) * 127.5, 0, 255).astype(np.uint8)
            png(os.path.join(ROOT, out), grid(px))
            print(f"wrote {out}")
            return
        train = allx
        leg = legibility(flat, train)
        rng = np.random.default_rng(0)
        base = legibility(rng.normal(size=(IMAGES, EDGE * EDGE)).astype(np.float32), train)
        print(f"legibility: samples {leg.mean():.3f} (min {leg.min():.3f})  "
              f"random baseline {base.mean():.3f}")

    out = "results/flowers/samples.png"
    for i, a in enumerate(sys.argv):
        if a == "--png" and i + 1 < len(sys.argv):
            out = sys.argv[i + 1]
    os.makedirs(os.path.dirname(os.path.join(ROOT, out)) or ".", exist_ok=True)
    px = np.clip((images + 1.0) * 127.5, 0, 255).astype(np.uint8)
    png(os.path.join(ROOT, out), grid(px))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
