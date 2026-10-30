"""The TEM backbone as a denoising score network.

Same patch embedding, blocks, RoPE, QK-RMSNorm, SwiGLU and adaLN; the energy
head is replaced by a per-patch score head. Training is denoising score
matching, which is first-order (no second-order gradient), so BendTorch can
train it. This is the denoising variant, not the paper's contrastive EBM.
"""
import torch
import torch.nn as nn

from tem_small import TEM


class ScoreTEM(TEM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.score = nn.Linear(self.pos.shape[-1], 3 * self.patch * self.patch)

    def forward(self, x, y):
        b = x.shape[0]
        z = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos
        cond = self.label(y)
        for block in self.blocks:
            z = block(z, cond)
        s = self.score(z)
        g, p = self.grid, self.patch
        return (s.reshape(b, g, g, 3, p, p)
                .permute(0, 3, 1, 4, 2, 5)
                .reshape(b, 3, g * p, g * p))
