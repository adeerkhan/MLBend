#!/usr/bin/env python3
"""Oxford Flowers at 64x64: RGB bytes in, one grayscale byte per pixel out.

  reference/.venv/bin/python reference/flowers_export.py

The pinned export `flowers64-512.u8` is 512 images of 64x64x3 interleaved RGB
(6,291,456 bytes; 12,288 per image). The flow model reads one byte per pixel and
one 64-byte row per token, so it needs one channel: this writes
`flowers64-512-gray.u8`, 512 images of 64x64 luma (2,097,152 bytes), and prints
the sha256 to pin.

Luma is BT.601 in integer arithmetic, (77 R + 150 G + 29 B) >> 8, so the file is
reproducible on any machine without a float rounding question.
"""
import hashlib
import os

import numpy as np

SRC = os.path.expanduser("~/.cache/bend-lab/flowers/flowers64-512.u8")
SRC_SHA = "02385f3ae6c9aa9c0cbe294b45e9758e347cb04abda529352952cd8b7b7075a8"
OUT = os.path.expanduser("~/.cache/bend-lab/flowers/flowers64-512-gray.u8")
IMAGES, EDGE = 512, 64


def main():
    blob = open(SRC, "rb").read()
    got = hashlib.sha256(blob).hexdigest()
    if got != SRC_SHA:
        raise SystemExit(f"{SRC} is {got}, not the pinned {SRC_SHA}")
    rgb = np.frombuffer(blob, dtype=np.uint8).reshape(IMAGES, EDGE, EDGE, 3).astype(np.uint32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    gray = ((77 * r + 150 * g + 29 * b) >> 8).astype(np.uint8)
    if gray.size != IMAGES * EDGE * EDGE:
        raise SystemExit(f"{gray.size} bytes, expected {IMAGES * EDGE * EDGE}")
    open(OUT, "wb").write(gray.tobytes())
    h = hashlib.sha256(gray.tobytes()).hexdigest()
    print(f"{OUT}")
    print(f"  {gray.size} bytes, {IMAGES} images of {EDGE}x{EDGE}")
    print(f"  mean {gray.mean():.1f}  min {gray.min()}  max {gray.max()}")
    print(f"  sha256 {h}")


if __name__ == "__main__":
    main()
