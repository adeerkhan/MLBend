"""The three-arm comparison, in one JSON and one page (milestone M4).

  reference/.venv/bin/python reference/compare_vit.py

The arms, on the *same* 1024 optdigits rows, the same starting cloud
(`noise_of(2000000, 4096)`) and the same step counts (64-step ancestral,
16-step DDIM) where each arm can be run that way:

  bend-vit        examples/diffusion_vit/main.bend — BendTorch layers only,
                  causal attention (ISSUE-011), pre-LN, one head
  torch-vit       the same architecture and hyperparameters in PyTorch
                  (reference/digits_vit_probe.py, HEADS=1 POS=features
                  CAUSAL=1 LR=3e-3 BATCH=64)
  lucidrains-unet the vendored reference UNet (reference/lucidrains_compare.py)

For every arm and both samplers it records: the loss (first, final), the
cloud's statistics, the legibility (mean nearest-digit |corr|) and the
nearest-digit MSE. The arms disagree about *architecture* — that is the
point of the comparison — so the JSON says what each one is, and the page
says where the numbers came from.

It reads what the arms' own scripts wrote; it never runs an arm itself.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "compare")
ARMS = (
    ("bend-vit", os.path.join(ROOT, "results", "diffusion-vit", "run.json"),
     "Bend, examples/diffusion_vit (causal attention, pre-LN, 100,736 params)"),
    ("torch-vit", os.path.join(OUT, "torch-vit.json"),
     "PyTorch, the same architecture and hyperparameters"),
    ("lucidrains-unet", os.path.join(ROOT, "results", "lucidrains-diffusion", "run.json"),
     "the vendored lucidrains UNet"),
)


def per_sampler(raw, arm):
    """Each arm's own shape, to one shape: {ddim, ddpm} -> numbers."""
    if arm == "lucidrains-unet":
        out = {}
        for k, v in (("ddim", raw.get("lucidrains", {}).get("ddim")),
                     ("ddpm", raw.get("lucidrains", {}).get("ancestral"))):
            if v:
                out[k] = v
        return out
    if "per_sampler" in raw:
        return raw["per_sampler"]
    return {}


def losses(raw, arm):
    if arm == "lucidrains-unet":
        return raw.get("loss", {})
    if arm == "bend-vit":
        return {"first": raw.get("first_loss"), "final": raw.get("final_loss")}
    return raw.get("loss", {})


def main():
    page = []
    data = {"task": "compare-vit", "data": "optdigits, first 1024 rows (sha256 pinned)",
            "starting_cloud": "noise_of(2000000, 4096)", "arms": {}}
    for arm, path, blurb in ARMS:
        if not os.path.exists(path):
            print(f"missing {os.path.relpath(path, ROOT)} — the {arm} arm has not been run")
            data["arms"][arm] = {"missing": os.path.relpath(path, ROOT), "what": blurb}
            continue
        raw = json.load(open(path))
        entry = {"what": blurb, "source": os.path.relpath(path, ROOT),
                 "loss": losses(raw, arm), "samplers": {}}
        for name, v in per_sampler(raw, arm).items():
            entry["samplers"][name] = {
                "steps": 16 if name == "ddim" else 64,
                "cloud": v.get("cloud"),
                "legibility": v.get("legibility"),
                "legibility_min": v.get("legibility_min"),
                "legibility_max": v.get("legibility_max"),
                "nearest_mse": v.get("nearest_mse"),
            }
        data["arms"][arm] = entry

    os.makedirs(OUT, exist_ok=True)
    json.dump(data, open(os.path.join(OUT, "compare.json"), "w"), indent=2)

    def cell(v, fmt="{:.3f}"):
        return "—" if v is None else fmt.format(v)

    rows = []
    for arm, entry in data["arms"].items():
        if "missing" in entry:
            rows.append(f"<tr><td>{arm}</td><td colspan=6 class=dim>not run "
                        f"({entry['missing']})</td></tr>")
            continue
        for name in ("ddpm", "ddim"):
            s = entry["samplers"].get(name)
            if not s:
                continue
            c = s.get("cloud") or {}
            rows.append(
                f"<tr><td>{arm}</td><td>{name} ({s['steps']} steps)</td>"
                f"<td class=num>{cell(entry['loss'].get('first'), '{:.4f}')}</td>"
                f"<td class=num>{cell(entry['loss'].get('final'), '{:.4f}')}</td>"
                f"<td class=num>{cell(s.get('legibility'))}</td>"
                f"<td class=num>{cell(s.get('nearest_mse'), '{:.4f}')}</td>"
                f"<td class=num>{cell(c.get('mean'), '{:+.3f}')} / {cell(c.get('std'))}"
                f" · {c.get('in_unit', '—')}/{c.get('n', '—')}</td></tr>")

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>three arms on the digits</title>
<style>
 body{{background:#0e1116;color:#e6e9ef;font:15px/1.5 ui-sans-serif,system-ui,sans-serif;
      margin:0;padding:32px}}
 h1{{font-size:20px;font-weight:600;margin:0 0 4px}}
 p{{color:#98a2b3;margin:4px 0 20px;max-width:70ch}}
 table{{border-collapse:collapse;width:100%;max-width:1000px}}
 th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid #222833}}
 th{{color:#98a2b3;font-weight:600;font-size:13px;text-transform:uppercase;
     letter-spacing:.04em}}
 td.num{{font-variant-numeric:tabular-nums;text-align:right}}
 .dim{{color:#667085}}
 .wl{{color:#5ee6c8}}
</style></head><body>
<h1>Three arms, one task, one starting cloud</h1>
<p>optdigits, the first 1024 rows; noise prediction, T = 64, linear schedule;
the same starting noise for every arm (<code>noise_of(2000000, 4096)</code>).
Legibility is the mean best absolute correlation of a sample with any training
digit (random images 0.326). The gate is 0.75. The Bend and PyTorch arms
attend causally — BendTorch's only attention is causal (ISSUE-011) — while the
reference UNet is convolutional; the architectures differ on purpose.</p>
<table>
<tr><th>arm</th><th>sampler</th><th>first loss</th><th>final loss</th>
<th>legibility</th><th>nearest-digit MSE</th><th>cloud mean / std · in [0,1]</th></tr>
{chr(10).join(rows)}
</table>
<p class=dim>Written by reference/compare_vit.py from each arm's own run.
Numbers, not claims: results/compare/compare.json.</p>
</body></html>
"""
    open(os.path.join(OUT, "index.html"), "w").write(html)
    print(f"wrote {os.path.relpath(OUT, ROOT)}/compare.json and index.html")
    for arm, entry in data["arms"].items():
        if "missing" in entry:
            print(f"  {arm}: not run")
            continue
        for name, s in entry["samplers"].items():
            print(f"  {arm:16s} {name}: legibility {s.get('legibility')}  "
                  f"mse {s.get('nearest_mse')}")


if __name__ == "__main__":
    sys.exit(main())
