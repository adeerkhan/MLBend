#!/usr/bin/env python3
"""What this machine can build, and which flow preset fits.

  scripts/system-profile.py
  scripts/system-profile.py --json

`bend -o`'s memory grows with the program, and the failure mode is the whole
machine swapping, not the compiler exiting. This prints the budget and the
preset that stays inside it, so the build is chosen before it starts.

Every cost below was measured on this repository inside a cgroup with swap off
(`scripts/bend-guard.sh`), sampling RSS once a second. A preset with no
measurement says so rather than guessing a number.
"""
import json
import os
import shutil
import subprocess
import sys

# preset -> (memory in GB, wall seconds, what it was, whether it built)
# A killed build's number is a floor, not a cost: it never counts as fitting.
MEASURED = {
    "tiny":   (7.5, 11, "over 7.5 GB, killed", False),
    "small":  (None, None, "not measured", False),
    "medium": (None, None, "not measured", False),
    "full":   (9.3, None, "9.3 GB and still climbing at 15 min, killed", False),
}
ORDER = ["tiny", "small", "medium", "full"]

# The measurements that say the cost is the program's shape, not the model's:
# a one-block 64-wide model costs more to compile than the 512-row three-block
# one did before the builders moved into the same file.
NOTES = [
    "the ViT example, the same architecture at 512 rows, builds in 82 s under 8 GB",
    "a one-block 64-wide model is already over 7.5 GB: the cost is the number of",
    "large recursive builders in one program, not the dimensions",
]


def mem_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1024 / 1024
    return 0.0


def gpu():
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                                  "--format=csv,noheader"], capture_output=True,
                                 text=True, timeout=5)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip().splitlines()[0]
        except Exception:
            pass
    return "none"


def main():
    ram = mem_gb()
    cores = os.cpu_count() or 1
    card = gpu()
    # the build is one process; leave the desktop and the page cache their share
    budget = min(0.55 * ram, 12.0)
    fit = [p for p in ORDER
           if MEASURED[p][3] and MEASURED[p][0] is not None and MEASURED[p][0] <= budget]
    info = {
        "ram_gb": round(ram, 1),
        "cores": cores,
        "gpu": card,
        "build_budget_gb": round(budget, 1),
        "fits": fit,
        "presets": {p: {"gb": MEASURED[p][0], "wall_s": MEASURED[p][1],
                        "status": MEASURED[p][2], "built": MEASURED[p][3]} for p in ORDER},
    }
    if "--json" in sys.argv:
        print(json.dumps(info, indent=2))
        return
    print(f"machine: {ram:.1f} GB RAM, {cores} cores, GPU {card}")
    print(f"build budget: {budget:.1f} GB (55% of RAM, capped at 12)")
    print("flow presets (compile memory, measured in a cgroup with swap off):")
    for p in ORDER:
        gb, wall, status, built = MEASURED[p]
        mark = "  <- fits" if p in fit else ""
        print(f"  {p:7s} {status}{mark}")
    if fit:
        print(f"\nrecommended: PRESET={fit[-1]}")
        print(f"  make flow64 PRESET={fit[-1]} MEM={budget:.0f}G")
    else:
        print("\nrecommended: none of them fit this machine yet")
        print("  make flow64 PRESET=tiny MEM=" + f"{budget:.0f}G  # dies at the cap, desktop safe")
        for n in NOTES:
            print(f"  {n}")


if __name__ == "__main__":
    main()
