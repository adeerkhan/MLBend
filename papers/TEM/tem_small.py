"""A 64x64 pixel-space TEM: the paper's energy backbone at a size that fits here.

The paper (arXiv 2610.31889) runs a class-conditional ViT energy over SD-VAE
latents at 256/512. This is the same energy structure in pixel space:

  patch embedding (8x8 conv), 2D RoPE, QK-RMSNorm, SwiGLU, shift-free adaLN
  from the class embedding, and energy heads at 1/3, 2/3 and 3/3 of the depth,
  summed into one scalar energy per image.

PRA (DINOv2 alignment), the KL term and the repulsion term are the paper's
scale components and are not in this first arm. The paper's FID numbers are
not comparable to this protocol.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope_half(x, pos, base=10000.0):
    """Rotary embedding on the last dim, using one integer position per token."""
    dim = x.shape[-1]
    half = dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, device=x.device, dtype=torch.float32) / half))
    ang = pos.float().unsqueeze(-1) * freqs
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    cos, sin = cos.unsqueeze(0).unsqueeze(2), sin.unsqueeze(0).unsqueeze(2)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def rope2d(x, grid):
    """Half of each head's channels rotates with the row, half with the column."""
    n = grid * grid
    rows = torch.arange(grid, device=x.device).repeat_interleave(grid)
    cols = torch.arange(grid, device=x.device).repeat(grid)
    d = x.shape[-1] // 2
    return torch.cat([rope_half(x[..., :d], rows), rope_half(x[..., d:], cols)], dim=-1)


class Attention(nn.Module):
    def __init__(self, dim, heads, grid):
        super().__init__()
        self.heads = heads
        self.grid = grid
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, n, self.heads, -1)
        k = k.view(b, n, self.heads, -1)
        v = v.view(b, n, self.heads, -1)
        q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + 1e-6)
        q, k = rope2d(q, self.grid), rope2d(k, self.grid)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, n, d)
        return self.proj(out)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.w12 = nn.Linear(dim, 2 * hidden)
        self.w3 = nn.Linear(hidden, dim)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


class Block(nn.Module):
    def __init__(self, dim, heads, grid, mlp_mult=4):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, heads, grid)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mlp_mult * dim)
        self.ada = nn.Linear(dim, 4 * dim)

    def forward(self, x, cond):
        scale1, gate1, scale2, gate2 = self.ada(cond).chunk(4, dim=-1)
        x = x + gate1.unsqueeze(1) * self.attn(self.norm1(x) * (1 + scale1.unsqueeze(1)))
        x = x + gate2.unsqueeze(1) * self.mlp(self.norm2(x) * (1 + scale2.unsqueeze(1)))
        return x


class EnergyHead(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # No bias is the CD stabiliser; the denoising regression wants one.
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(),
                                 nn.Linear(dim, 1, bias=bias))

    def forward(self, x):
        return self.mlp(self.norm(x.mean(1))).squeeze(-1)


class TEM(nn.Module):
    def __init__(self, depth=6, dim=192, heads=6, patch=8, grid=8,
                 in_ch=3, num_classes=102, head_layers=None, pra_layer=1,
                 head_bias=False):
        super().__init__()
        self.depth = depth
        self.pra_layer = pra_layer
        self.patch = patch
        self.grid = grid
        self.patch_embed = nn.Conv2d(in_ch, dim, patch, patch)
        self.pos = nn.Parameter(torch.zeros(1, grid * grid, dim))
        self._init_pos()
        self.label = nn.Embedding(num_classes, dim)
        self.blocks = nn.ModuleList([Block(dim, heads, grid) for _ in range(depth)])
        layers = head_layers or tuple((j * depth) // 3 - 1 for j in (1, 2)) + (depth - 1,)
        self.head_layers = tuple(layers)
        self.heads = nn.ModuleList([EnergyHead(dim, head_bias) for _ in self.head_layers])
        self.projector = nn.Conv2d(dim, 384, 3, padding=1)
        self.apply(self._init_weights)

    def _init_pos(self):
        # A learned position table; the 2D structure comes from the RoPE in
        # attention, so this starts small like the paper's learned table.
        nn.init.trunc_normal_(self.pos, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, x, y, pra_layer=None):
        b = x.shape[0]
        z = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos
        cond = self.label(y)
        outs, pra = [], None
        hi = 0
        for i, block in enumerate(self.blocks):
            z = block(z, cond)
            if pra_layer is not None and i == pra_layer:
                p = z.transpose(1, 2).reshape(b, self.pos.shape[-1], self.grid, self.grid)
                pra = self.projector(p).flatten(2).transpose(1, 2)
            if hi < len(self.head_layers) and i == self.head_layers[hi]:
                outs.append(self.heads[hi](z))
                hi += 1
                if hi == len(self.head_layers):
                    break
        e = torch.stack(outs, dim=1)
        return (e, pra) if pra_layer is not None else e

    def energy(self, x, y):
        return self(x, y).sum(dim=1)


@torch.no_grad()
def count_params(model):
    return sum(p.numel() for p in model.parameters())


def langevin(model, x, y, steps, step_size, noise_std, clamp=1.0):
    """The paper's ordering: add noise, take the gradient step, project."""
    model.eval()
    for _ in range(steps):
        with torch.enable_grad():
            x = x + torch.randn_like(x) * noise_std
            x = x.detach().requires_grad_(True)
            e = model.energy(x, y).sum()
            g = torch.autograd.grad(e, x)[0]
        x = (x - step_size * g).detach().clamp(-clamp, clamp)
    return x
