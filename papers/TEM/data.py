"""Oxford Flowers 102, pinned and reduced to 64x64.

  reference/.venv-lucidrains/bin/python papers/TEM/data.py --build

The source is the HuggingFace mirror of the official Oxford Flowers 102 set
(`nelorth/oxford-flowers`): 7169 train and 1020 test images, 102 classes. The
two parquet files are pinned by SHA-256; the 64x64 arrays are derived once and
cached under `~/.cache/bend-lab/flowers` together with their own hashes.

The 64x64 reduction is ours, not the paper's: TEM trains on 256x256 SD-VAE
latents. A pixel-space 64x64 protocol is what fits this repository, and the
paper's FID numbers are not comparable to it.
"""
import hashlib
import io
import json
import os
import sys
import urllib.request

import numpy as np
from PIL import Image
import pyarrow.parquet as pq

CACHE = os.path.expanduser("~/.cache/bend-lab/flowers")
BASE = "https://huggingface.co/datasets/nelorth/oxford-flowers/resolve/main/data/"
FILES = {
    "train": ("train-00000-of-00001-12de94e121bdbead.parquet",
              "f53b6c9f81c492a0b897006d794bcb485f10022e9cc953a11632d2f91a61482e"),
    "test": ("test-00000-of-00001-96eeec628415add6.parquet",
             "f7582775856bfbb59f4bffed920b0977ebd1722deea93d6200e840732b92e4a0"),
}
SIZE = 64


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(split):
    name, want = FILES[split]
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        os.makedirs(CACHE, exist_ok=True)
        print("fetching", name)
        urllib.request.urlretrieve(BASE + name, path)
    got = sha256(path)
    if got != want:
        raise SystemExit(f"{name}: sha256 {got} != {want}")
    return name, path


def build():
    os.makedirs(CACHE, exist_ok=True)
    manifest = {"source": "nelorth/oxford-flowers", "size": SIZE, "splits": {}}
    for split in ("train", "test"):
        name, path = fetch(split)
        table = pq.read_table(path, columns=["image", "label"])
        rows = table.to_pylist()
        images = np.zeros((len(rows), SIZE, SIZE, 3), dtype=np.uint8)
        labels = np.zeros((len(rows),), dtype=np.int64)
        for i, row in enumerate(rows):
            im = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
            images[i] = np.asarray(im.resize((SIZE, SIZE), Image.Resampling.BICUBIC))
            labels[i] = row["label"]
        out = os.path.join(CACHE, f"flowers{SIZE}-{split}.npz")
        np.savez(out, images=images, labels=labels)
        manifest["splits"][split] = {
            "parquet": name,
            "parquet_sha256": FILES[split][1],
            "npz": os.path.basename(out),
            "npz_sha256": sha256(out),
            "count": int(len(rows)),
            "classes": int(labels.max()) + 1,
        }
        print(split, len(rows), "images", images.shape, sha256(out))
    with open(os.path.join(CACHE, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(json.dumps(manifest, indent=2))


def load(split="train"):
    path = os.path.join(CACHE, f"flowers{SIZE}-{split}.npz")
    if not os.path.exists(path):
        raise SystemExit(f"{path} is missing: run --build first")
    d = np.load(path)
    return d["images"], d["labels"]


if __name__ == "__main__":
    if "--build" in sys.argv:
        build()
    else:
        for s in ("train", "test"):
            x, y = load(s)
            print(s, x.shape, x.dtype, "labels", y.min(), y.max(), "classes", len(set(y.tolist())))
