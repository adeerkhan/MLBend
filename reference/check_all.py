"""Runs every MLBend verification. Exit code 0 = everything is fine.

  reference/.venv/bin/python reference/check_all.py
"""
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, "reference/.venv/bin/python")
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
BEND_PIN = os.environ.get("BEND_PIN") or "2.0.37"
ENV = dict(os.environ, BEND_NO_TELEMETRY="1",
           PATH=os.path.expanduser("~/.elan/bin") + ":" + os.environ["PATH"])
results = []


def run(name, cmd, ok_if=None, timeout=1800):
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=ENV, timeout=timeout)
    out = r.stdout + r.stderr
    ok = (r.returncode == 0) if ok_if is None else ok_if(out, r.returncode)
    results.append(ok)
    print(f"{'ok  ' if ok else 'ERROR'} {name} ({time.time() - t0:.0f}s)")
    if not ok:
        print("     " + out.strip().replace("\n", "\n     ")[-600:])
    return out


def check_toolchain():
    """Fail early, with a clear message, when the pinned toolchain is not the one in use."""
    bv = subprocess.run(BEND.split() + ["version"], capture_output=True, text=True, env=ENV).stdout.split()
    if not bv or bv[-1] != BEND_PIN:
        sys.exit(f"ERROR: bend {BEND_PIN} is required (found {' '.join(bv) or 'none'}); run scripts/setup.sh")
    lv = subprocess.run(["lean", "--version"], capture_output=True, text=True, env=ENV)
    if "version 4.34.0" not in lv.stdout:
        sys.exit(f"ERROR: lean 4.34.0 is required for --verdict (found {lv.stdout.strip() or 'none'}); run scripts/setup.sh")


def main():
    check_toolchain()
    packages = sorted(d for d in os.listdir(ROOT)
                      if os.path.isdir(os.path.join(ROOT, d)) and os.path.exists(os.path.join(ROOT, d, "main.bend")))

    # 1. the proofs: the checker, then the audited kernel
    for pkg in packages:
        f = f"{pkg}/main.bend"
        run(f"{f}: ALL PROOFS CHECK", BEND.split() + [f],
            lambda o, c: "ALL PROOFS CHECK" in o and "FAIL" not in o)
        run(f"{f} --verdict (the kernel, as Lean)", BEND.split() + [f, "--verdict"],
            lambda o, c: o.strip().endswith("ALL PROOFS CHECK"))

    # 2. the packages with laws carry no @unsafe and no ?TODO
    for pkg in packages:
        f = os.path.join(ROOT, pkg, "main.bend")
        src = open(f).read()
        clean = not re.search(r"@unsafe|^\s*\?\w+|sorry", re.sub(r"#.*", "", src), re.M)
        results.append(clean)
        print(f"{'ok  ' if clean else 'ERROR'} {pkg}/main.bend: no @unsafe or ?TODO")

    # 3. the programs that must be refused
    for pkg in packages:
        tdir = os.path.join(ROOT, pkg, "tests")
        if not os.path.isdir(tdir):
            continue
        for name in sorted(os.listdir(tdir)):
            if not name.startswith("bad_") or not name.endswith(".bend"):
                continue
            f = f"{pkg}/tests/{name}"
            out = run(f"{f}: refused", BEND.split() + [f, "--check-only"],
                      lambda o, c: "SOME PROOFS FAIL" in o and "Error:" in o)
            results.append("MUST FAIL" in out)

    # 4. the programs that must run
    for pkg in packages:
        tdir = os.path.join(ROOT, pkg, "tests")
        if not os.path.isdir(tdir):
            continue
        for name in sorted(os.listdir(tdir)):
            if name.startswith("bad_") or not name.endswith(".bend"):
                continue
            f = f"{pkg}/tests/{name}"
            run(f"{f}: runs", BEND.split() + [f], lambda o, c: "SOME PROOFS FAIL" not in o and "Error:" not in o)

    # 5. the numbers, against NumPy and scikit-learn
    for name in sorted(os.listdir(os.path.join(ROOT, "reference"))):
        if not (name.startswith("test_") and name.endswith(".py")):
            continue
        run(f"reference/{name}: the numbers match", [PY, os.path.join("reference", name)])

    ok = sum(results)
    print(f"\n{ok}/{len(results)} checks passed")
    sys.exit(0 if ok == len(results) else 1)


if __name__ == "__main__":
    main()
