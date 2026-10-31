#!/usr/bin/env python3
"""One grayscale byte per pixel, at 64 or 256, from the pinned flowers data.

  reference/.venv-lucidrains/bin/python reference/flowers_export.py --size 256

64: converts the pinned 64x64 RGB export (`flowers64-512.u8`, 512 images of
64x64x3 interleaved) to luma. The flow's tokeniser reads one byte per pixel and
one row per token, so RGB would have had it reading a scrambled raster.

256: decodes the first 512 images of the pinned Oxford Flowers parquet at
256x256. This is the source the 64x64 export was reduced from, so the two data
sets are the same images at two resolutions, not one upscaled.

Luma is BT.601 in integer arithmetic, (77 R + 150 G + 29 B) >> 8, so both files
are reproducible without a float rounding question.
"""
import hashlib
import io
import os
import sys

import numpy as np

CACHE = os.path.expanduser("~/.cache/bend-lab/flowers")

SRC64 = os.path.join(CACHE, "flowers64-512.u8")
SRC64_SHA = "02385f3ae6c9aa9c0cbe294b45e9758e347cb04abda529352952cd8b7b7075a8"

PQ = os.path.join(CACHE, "train-00000-of-00001-12de94e121bdbead.parquet")
PQ_SHA = "f53b6c9f81c492a0b897006d794bcb485f10022e9cc953a11632d2f91a61482e"

N = 512


def luma(rgb):
    r, g, b = rgb[..., 0].astype(np.uint32), rgb[..., 1].astype(np.uint32), rgb[..., 2].astype(np.uint32)
    return ((77 * r + 150 * g + 29 * b) >> 8).astype(np.uint8)


def from64():
    blob = open(SRC64, "rb").read()
    got = hashlib.sha256(blob).hexdigest()
    if got != SRC64_SHA:
        raise SystemExit(f"{SRC64} is {got}, not the pinned {SRC64_SHA}")
    rgb = np.frombuffer(blob, dtype=np.uint8).reshape(N, 64, 64, 3)
    return luma(rgb), 64


def from256():
    import pyarrow.parquet as pq
    from PIL import Image

    blob = open(PQ, "rb").read()
    got = hashlib.sha256(blob).hexdigest()
    if got != PQ_SHA:
        raise SystemExit(f"{PQ} is {got}, not the pinned {PQ_SHA}")
    rows = pq.read_table(PQ, columns=["image"]).to_pylist()[:N]
    out = np.zeros((N, 256, 256), dtype=np.uint8)
    for i, row in enumerate(rows):
        im = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        im = im.resize((256, 256), Image.Resampling.BICUBIC)
        out[i] = luma(np.asarray(im))
    return out, 256


def split():
    """40 training and 1 held-out image per class, evenly across the 102 classes.

    The parquet is grouped by label, so a contiguous holdout is one class (the
    first 512 images are classes 0..11). This takes each class's first image as
    the holdout and its next 40 as training.
    """
    import io as _io

    import pyarrow.parquet as pq
    from PIL import Image

    blob = open(PQ, "rb").read()
    got = hashlib.sha256(blob).hexdigest()
    if got != PQ_SHA:
        raise SystemExit(f"{PQ} is {got}, not the pinned {PQ_SHA}")
    rows = pq.read_table(PQ, columns=["image", "label"]).to_pylist()
    by_class = {}
    for r in rows:
        by_class.setdefault(r["label"], []).append(r["image"]["bytes"])
    train, hold = [], []
    for c in sorted(by_class):
        imgs = by_class[c]
        if len(imgs) < 2:
            continue
        hold.append(imgs[0])
        train.extend(imgs[1:41])
    def gray_of(jpegs, name):
        out = np.zeros((len(jpegs), 64, 64), dtype=np.uint8)
        for i, b in enumerate(jpegs):
            im = Image.open(_io.BytesIO(b)).convert("RGB").resize((64, 64), Image.Resampling.BICUBIC)
            out[i] = luma(np.asarray(im))
        path = os.path.join(CACHE, name)
        data = out.tobytes()
        open(path, "wb").write(data)
        print(path)
        print(f"  {len(data)} bytes, {len(jpegs)} images of 64x64, classes {len(by_class)}")
        print(f"  sha256 {hashlib.sha256(data).hexdigest()}")
    print(f"classes {len(by_class)}, train {len(train)}, hold {len(hold)}")
    gray_of(train, "flowers64-train-gray.u8")
    gray_of(hold, "flowers64-hold-gray.u8")


def main():
    size = 64
    if "--split" in sys.argv:
        split()
        return
    if "--size" in sys.argv:
        size = int(sys.argv[sys.argv.index("--size") + 1])
    gray, size = from64() if size == 64 else from256()
    out = os.path.join(CACHE, f"flowers{size}-512-gray.u8")
    data = gray.tobytes()
    open(out, "wb").write(data)
    print(out)
    print(f"  {len(data)} bytes, {N} images of {size}x{size}")
    print(f"  mean {gray.mean():.1f}  min {gray.min()}  max {gray.max()}")
    print(f"  sha256 {hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
