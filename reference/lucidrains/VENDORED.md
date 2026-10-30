# Vendored: lucidrains/denoising-diffusion-pytorch

This directory is a **verbatim copy** of
[`lucidrains/denoising-diffusion-pytorch`](https://github.com/lucidrains/denoising-diffusion-pytorch)
at commit

```
c8155b1dc622e3f1ef0d53dc6e5a570ca105bcec   2026-09-29   "fix self cond in 1d diffusion"
```

cloned on 2026-10-27. Nothing here is modified: the hashes below are the
upstream bytes, and `git diff` on this directory must stay empty except for
these two files (`VENDORED.md`, `README.md`) and `requirements.txt`.

## License

MIT, Copyright (c) 2020 Phil Wang — the upstream `LICENSE` is copied beside
this file and applies to everything under `denoising_diffusion_pytorch/`.

## Why the whole package, not "the necessary files"

The upstream `denoising_diffusion_pytorch/__init__.py` imports **every**
submodule (classifier-free guidance, elucidated diffusion, the Karras Unets,
the 1d and 3d variants, `xm`). Importing `denoising_diffusion_pytorch` — which
is how the library is meant to be used — therefore executes all of them, so a
subset would not import unchanged. The packages needed by the core DDPM path
(`denoising_diffusion_pytorch.py` → `attend.py`, `version.py`) are a subset of
what `__init__` pulls, not a replacement for it.

## Files (sha256, upstream)

```
8ef3def059362c1e2f54d4d2c1fb56d3d7202c9f26131daba90da9651cee4f8c  denoising_diffusion_pytorch/attend.py
bd4d22425645cbd722451d3693a300bf339de57ccf6c66786f35e3889a3ae120  denoising_diffusion_pytorch/classifier_free_guidance.py
c68b696cf09ee563675643f656608918bb79ffd7c524b8c9c6ed7c76f18aab70  denoising_diffusion_pytorch/continuous_time_gaussian_diffusion.py
64ebc4732ec0ff50ebb839a350ca9b4f79bd4a5882a032aebf860bb2fe6c0962  denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
da4090859a5e6b480d6bb0c04f2db873369296b43f9887ca14c1ebf0b21bf989  denoising_diffusion_pytorch/denoising_diffusion_pytorch_1d.py
57b29e5f8b393f0829d872388edc21abf0dc3167bb0fda6eb41637a03d0f5e7b  denoising_diffusion_pytorch/elucidated_diffusion.py
3952960d132975bb05d6f15ad7b1babbbebda059e2132dd809da49c240162961  denoising_diffusion_pytorch/fid_evaluation.py
fe20875d6ed415f124661088fde2bfd03f1f92319cc4fd98e6c364b51c497d4c  denoising_diffusion_pytorch/guided_diffusion.py
753d1017e50f0d2d7d0b56d12e602ca80a3abf0de7f10401df6b9fb8eb64edb3  denoising_diffusion_pytorch/__init__.py
d89d4f5dd8868e1d2b861803a648eb67127d566fabcbc0e121b5ea7450217071  denoising_diffusion_pytorch/karras_unet_1d.py
79bc8a93940c23de435328326e8bcff39cdebeaeafc0dd03cea5ab0d87d026c5  denoising_diffusion_pytorch/karras_unet_3d.py
05bed8b42b9d69dd1b34ccd4a4e2df1e0a7acb25781255037957f95565566971  denoising_diffusion_pytorch/karras_unet.py
5c579af23b564ff86ae7106e698cc2103808983023000840e32f816632f515f2  denoising_diffusion_pytorch/learned_gaussian_diffusion.py
09b0eb4483fa76f51fdba27555a6c43d62667c173c47636da222cd63d98f80c0  denoising_diffusion_pytorch/repaint.py
2e2fba0d870317212560d8624a7022c5243de9718df3db9958e8e9dc9943268d  denoising_diffusion_pytorch/simple_diffusion.py
75020c5570c2b5fb400d54ed271487a513a718ae57592b8d6069a015732e9b7b  denoising_diffusion_pytorch/version.py
647588f16ec28aab91ba57e5ab779b138da70e42ea94f4f834d47e2e69608116  denoising_diffusion_pytorch/v_param_continuous_time_gaussian_diffusion.py
e6f278d0e9972741bace1b7709c1730b49b64dba5f123c65aed61b4bd824f333  denoising_diffusion_pytorch/weighted_objective_gaussian_diffusion.py
8a4e10ba8aec6f2f789476096cbb2c5bb2d13f0de8a8d87bcb192c422190a8df  denoising_diffusion_pytorch/xm.py
02cf6ed7df17f94fafa61e67a089ee51faac006db35ffa173c70e6763c253cf6  LICENSE
```

## How to check the copy

```console
$ cd reference/lucidrains && sha256sum -c VENDORED.sha256
```

`VENDORED.sha256` is the same list in `sha256sum -c` form.

## Note on `__version__`

The package's `__init__.py` does not re-export `version.py`'s `__version__`,
so `import denoising_diffusion_pytorch as d; d.__version__` raises
`AttributeError` (ISSUE-004 in the local issue ledger). The copy stays verbatim;
provenance is the commit above, not that attribute.
