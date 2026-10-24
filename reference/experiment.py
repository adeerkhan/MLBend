"""The experiment runner: one Bend program, one reference, one report card.

  reference/.venv/bin/python reference/experiment.py --list
  reference/.venv/bin/python reference/experiment.py --task iris-crossval
  reference/.venv/bin/python reference/experiment.py --all --wandb

This is the layer MLBend exists for. A **task** is a dataset, a program and
the numbers that program must produce; an **adapter** runs the program and
parses its output; a **reference** computes the same numbers with NumPy or
scikit-learn; and a **manifest** records everything a second person needs to
re-run it: the git commit, the Bend version, the package versions and hub
hashes, the machine, the seed and the tolerances.

The report is written to `reports/` as JSON. With `--wandb` the same manifest
is logged to Weights & Biases (project `mlbend`), offline when there is no
API key, so a run never blocks on the network.
"""
import argparse
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
import hashlib
import io
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, "reference/.venv/bin/python")
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
if os.path.isdir(ENV_DIR):
    ENV["BEND_LIB"] = ENV_DIR

SCHEMA = "mlbend.experiment/1"


# --------------------------------------------------------------------------
# the tasks
# --------------------------------------------------------------------------

def parse_iris_crossval(out):
    folds = [float(m) for m in re.findall(r"^fold \d+: ([0-9.eE+-]+)$", out, re.M)]
    mean = float(re.search(r"mean accuracy: ([0-9.eE+-]+)", out).group(1))
    return {"mean_accuracy": mean, "fold_min": min(folds), "fold_max": max(folds)}


def reference_iris_crossval():
    sys.path.insert(0, os.path.join(ROOT, "reference"))
    from test_crossval import load, sklearn_folds

    x, y = load()
    folds = sklearn_folds(x, y)
    return {"mean_accuracy": float(np.mean(folds)), "fold_min": min(folds), "fold_max": max(folds)}


def parse_iris_mlp(out):
    return {
        "loss_before": float(re.search(r"loss before: ([0-9.eE+-]+)", out).group(1)),
        "loss_after": float(re.search(r"loss after:  ([0-9.eE+-]+)", out).group(1)),
        "accuracy": float(re.search(r"accuracy: ([0-9.eE+-]+)", out).group(1)),
    }


def reference_iris_mlp():
    sys.path.insert(0, os.path.join(ROOT, "reference"))
    from test_mlp import sklearn_result

    # the reference trains, so its loss is not comparable; only the score is
    return {"accuracy": sklearn_result()}



    return {"mse_after": sklearn_mse()}


def parse_wine_kmeans(out):
    inertia = float(re.search(r"inertia: ([0-9.eE+-]+)", out).group(1))
    sizes = [int(n) for n in re.search(r"cluster sizes: ([0-9 ]+)", out).group(1).split()]
    return {"inertia": inertia, "cluster_sizes": sizes}


def reference_wine_kmeans():
    sys.path.insert(0, os.path.join(ROOT, "reference"))
    from kmeans_wine import kmeans, load, standardise

    inertia, sizes = kmeans(standardise(load()), 3, 10)
    return {"inertia": inertia, "cluster_sizes": sizes}


def parse_cancer_knn(out):
    return {"accuracy": float(re.search(r"accuracy: ([0-9.eE+-]+)", out).group(1))}


def parse_attention(out):
    return {"checksum": float(re.search(r"checksum: ([0-9.eE+-]+)", out).group(1))}


def parse_transformer(out):
    losses = [float(v) for v in re.findall(r"^loss: ([0-9.eE+-]+)$", out, re.M) if float(v) > 0.0]
    accuracy = float(re.search(r"^accuracy: ([0-9.eE+-]+)$", out, re.M).group(1))
    return {"first_loss": losses[0], "final_loss": losses[-1], "accuracy": accuracy}


def parse_cancer_nb(out):
    return {"accuracy": float(re.search(r"accuracy: ([0-9.eE+-]+)", out).group(1))}


def parse_heart_nb(out):
    return {"accuracy": float(re.search(r"accuracy: ([0-9.eE+-]+)", out).group(1))}


def parse_wine_pca(out):
    return {"explained_variance_ratio": float(re.search(r"explained variance ratio: ([0-9.eE+-]+)", out).group(1))}


def reference_cancer_knn():
    sys.path.insert(0, os.path.join(ROOT, "reference"))
    from test_knn import sklearn_knn

    return {"accuracy": sklearn_knn()}


def reference_attention():
    sys.path.insert(0, os.path.join(ROOT, "reference"))
    from test_attention import expected

    return {"checksum": float(expected().sum(dtype="float32"))}


def reference_transformer():
    sys.path.insert(0, os.path.join(ROOT, "reference"))
    from test_transformer import numpy_curve

    curve, final, accuracy = numpy_curve()
    return {"first_loss": float(curve[0]), "final_loss": final, "accuracy": accuracy}


def reference_cancer_nb():
    sys.path.insert(0, os.path.join(ROOT, "reference"))
    from test_naive_bayes import sklearn_result

    return {"accuracy": sklearn_result()}


def reference_heart_nb():
    sys.path.insert(0, os.path.join(ROOT, "reference"))
    from sklearn.naive_bayes import GaussianNB
    from test_naive_bayes_heart import load

    x, y = load()
    split = len(x) * 4 // 5
    return {"accuracy": float(GaussianNB().fit(x[:split], y[:split]).score(x[split:], y[split:]))}


def reference_wine_pca():
    sys.path.insert(0, os.path.join(ROOT, "reference"))
    from test_pca import sklearn_ratio

    return {"explained_variance_ratio": sklearn_ratio()}


TASKS = {
    "iris-crossval": {
        "description": "logistic regression over iris, five stratified folds, scaler fitted on the training rows only",
        "program": "examples/crossval/main.bend",
        "dataset": {"name": "iris", "rows": 150, "features": 4,
                    "source": "https://huggingface.co/datasets/mstz/iris",
                    "files": ["https://huggingface.co/datasets/mstz/iris/resolve/main/iris.data"]},
        "seed": 42,
        "parse": parse_iris_crossval,
        "reference": reference_iris_crossval,
        "metrics": {
            "mean_accuracy": {"tolerance": 0.03, "mode": "absolute"},
            # the two libraries shuffle with different generators, so a fold's
            # own score is not comparable; the mean is
            "fold_min": {"tolerance": 0, "mode": "report"},
            "fold_max": {"tolerance": 0, "mode": "report"},
        },
    },
    "iris-mlp": {
        "description": "a two-layer rectified network over iris, 400 SGD steps",
        "program": "examples/mlp/main.bend",
        "dataset": {"name": "iris", "rows": 150, "features": 4,
                    "source": "https://huggingface.co/datasets/mstz/iris",
                    "files": ["https://huggingface.co/datasets/mstz/iris/resolve/main/iris.data"]},
        "seed": 0,
        "parse": parse_iris_mlp,
        "reference": reference_iris_mlp,
        "metrics": {
            "accuracy": {"tolerance": 0.06, "mode": "absolute"},
        },
    },
    "wine-pca": {
        "description": "first principal component of wine, by power iteration",
        "program": "examples/pca/main.bend",
        "dataset": {"name": "wine", "rows": 6497, "features": 11,
                    "source": "https://huggingface.co/datasets/mstz/wine",
                    "files": ["https://huggingface.co/datasets/mstz/wine/resolve/main/wine/train.csv"]},
        "seed": 0,
        "parse": parse_wine_pca,
        "reference": reference_wine_pca,
        "metrics": {
            "explained_variance_ratio": {"tolerance": 0.001, "mode": "absolute"},
        },
    },
    "wine-kmeans": {
        "description": "k-means over wine, three clusters for ten rounds",
        "program": "examples/kmeans/main.bend",
        "dataset": {"name": "wine", "rows": 6497, "features": 11,
                    "source": "https://huggingface.co/datasets/mstz/wine",
                    "files": ["https://huggingface.co/datasets/mstz/wine/resolve/main/wine/train.csv"]},
        "seed": 0,
        "parse": parse_wine_kmeans,
        "reference": reference_wine_kmeans,
        "metrics": {
            "inertia": {"tolerance": 0.001, "mode": "relative"},
            "cluster_sizes": {"tolerance": 0, "mode": "exact"},
        },
    },
    "transformer-periodic": {
        "description": "a one-block causal transformer trained on a periodic sequence, BendTorch against NumPy",
        "program": "examples/transformer/main.bend",
        "dataset": {"name": "periodic-0-7", "rows": 32, "features": 8,
                    "source": "generated: two sequences of sixteen positions, vocab 8"},
        "seed": 0,
        "parse": parse_transformer,
        "reference": reference_transformer,
        "metrics": {
            "first_loss": {"tolerance": 0.25, "mode": "absolute"},
            "final_loss": {"tolerance": 0, "mode": "report"},
            "accuracy": {"tolerance": 0.1, "mode": "absolute"},
        },
    },
    "attention-parity": {
        "description": "single-head self-attention forward, four tokens of eight features",
        "program": "examples/attention/main.bend",
        "dataset": {"name": "synthetic-attention", "rows": 4, "features": 8,
                    "source": "generated by the formula in the program"},
        "seed": 0,
        "parse": parse_attention,
        "reference": reference_attention,
        "metrics": {
            "checksum": {"tolerance": 0.0001, "mode": "absolute"},
        },
    },
    "cancer-naive-bayes": {
        "description": "Gaussian naive Bayes over breast cancer, four fifths train",
        "program": "examples/naive_bayes/main.bend",
        "dataset": {"name": "breast_cancer", "rows": 683, "features": 9,
                    "source": "https://huggingface.co/datasets/mstz/breast",
                    "files": ["https://huggingface.co/datasets/mstz/breast/resolve/main/cancer/train.csv"]},
        "seed": 0,
        "parse": parse_cancer_nb,
        "reference": reference_cancer_nb,
        "metrics": {
            "accuracy": {"tolerance": 0.01, "mode": "absolute"},
        },
    },
    "heart-naive-bayes": {
        "description": "Gaussian naive Bayes over heart failure, four fifths train",
        "program": "examples/naive_bayes_heart/main.bend",
        "dataset": {"name": "heart_failure", "rows": 299, "features": 12,
                    "source": "https://huggingface.co/datasets/mstz/heart_failure",
                    "files": ["https://huggingface.co/datasets/mstz/heart_failure/resolve/main/death/train.csv"]},
        "seed": 0,
        "parse": parse_heart_nb,
        "reference": reference_heart_nb,
        "metrics": {
            "accuracy": {"tolerance": 0.02, "mode": "absolute"},
        },
    },
    "cancer-knn": {
        "description": "three-nearest-neighbours over breast cancer, four fifths train",
        "program": "examples/knn/main.bend",
        "dataset": {"name": "breast_cancer", "rows": 683, "features": 9,
                    "source": "https://huggingface.co/datasets/mstz/breast",
                    "files": ["https://huggingface.co/datasets/mstz/breast/resolve/main/cancer/train.csv"]},
        "seed": 0,
        "parse": parse_cancer_knn,
        "reference": reference_cancer_knn,
        "metrics": {
            "accuracy": {"tolerance": 0.02, "mode": "absolute"},
        },
    },
}


# --------------------------------------------------------------------------
# running a task
# --------------------------------------------------------------------------

def binary_for(task):
    return os.path.join(tempfile.gettempdir(), "mlbend-exp-" + task)


def run_bend(task, cfg, repeat=1):
    """Build and run the task's program.

    Returns the first run's output, the build time, every run's wall time, and
    the minimum of them. The build is reported separately: a Bend program is
    compiled, and the reference is not.
    """
    binary = binary_for(task)
    start = time.perf_counter()
    build = subprocess.run([BEND, "-o", binary, cfg["program"]],
                           capture_output=True, text=True, env=ENV, cwd=ROOT)
    build_seconds = time.perf_counter() - start
    if build.returncode != 0:
        raise SystemExit(f"{task}: the program did not build:\n{build.stdout}{build.stderr}")
    times, output = [], None
    for _ in range(max(1, repeat)):
        start = time.perf_counter()
        out = subprocess.run([binary], capture_output=True, text=True, env=ENV, cwd=ROOT)
        times.append(time.perf_counter() - start)
        if out.returncode != 0:
            raise SystemExit(f"{task}: the program did not run:\n{out.stdout}{out.stderr}")
        output = out.stdout
    return output, build_seconds, times, min(times)


def compare(cfg, bend_values, reference_values):
    """The metric table: the Bend number, the reference number, and the verdict."""
    table = {}
    for name, rule in cfg["metrics"].items():
        bend = bend_values.get(name)
        want = reference_values.get(name)
        if bend is None:
            table[name] = {"bend": None, "reference": want, "pass": False,
                           "note": "the program did not print this number"}
            continue
        if want is None:
            table[name] = {"bend": bend, "reference": None, "pass": True,
                           "note": "no reference for this number"}
            continue
        if rule["mode"] == "report":
            table[name] = {"bend": bend, "reference": want, "pass": True,
                           "note": "different folds; reported, not compared"}
            continue
        if rule["mode"] == "exact":
            delta = 0 if bend == want else None
            ok = bend == want
        else:
            delta = bend - want
            limit = rule["tolerance"] * abs(want) if rule["mode"] == "relative" else rule["tolerance"]
            ok = abs(delta) <= limit
        table[name] = {"bend": bend, "reference": want, "delta": delta,
                       "tolerance": rule["tolerance"], "mode": rule["mode"], "pass": ok}
    return table


def cache_file(url):
    """The cache path of a HuggingFace file URL, laid out as <repo>/<path>."""
    m = re.search(r"/datasets/(.+?)/resolve/[^/]+/(.+)$", url)
    if not m:
        return None
    return os.path.join(ROOT, "reference/.cache/datasets", m.group(1), m.group(2))


def dataset_probe(cfg):
    """Hash the dataset bytes, fill the local cache, and time the fetch once."""
    dataset = cfg["dataset"]
    files = dataset.get("files", [])
    cache_root = os.path.join(ROOT, "reference/.cache/datasets")
    ENV["MLBEND_DATA"] = cache_root
    start = time.perf_counter()
    digest = hashlib.sha256()
    total = 0
    if files:
        for url in files:
            path = cache_file(url)
            if path and os.path.isfile(path):
                data = open(path, "rb").read()
            else:
                data = urllib.request.urlopen(url, timeout=120).read()
                if path:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "wb") as f:
                        f.write(data)
            digest.update(data)
            total += len(data)
    else:
        digest.update((dataset.get("name", "") + "\n" + dataset.get("source", "")).encode())
    seconds = time.perf_counter() - start
    return {
        "files": files,
        "bytes": total,
        "sha256": digest.hexdigest(),
        "fetch_seconds": round(seconds, 3),
        "cache": cache_root,
    }


def git_state():
    def run(*args):
        r = subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True)
        return r.stdout.strip()

    return {"commit": run("rev-parse", "--short", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain"))}


def bend_version():
    try:
        return subprocess.run([BEND, "version"], capture_output=True, text=True,
                              env=ENV, timeout=30).stdout.split()[-1]
    except Exception:
        return None


def packages():
    """The published packages in bend.lock: name, version and hub hash."""
    path = os.path.join(ROOT, "bend.lock")
    if not os.path.isfile(path):
        return []
    out, block = [], {}
    for line in open(path):
        line = line.strip()
        if line == "[[package]]":
            if block.get("name"):
                out.append(block)
            block = {}
        elif "=" in line and not line.startswith("["):
            key, value = [s.strip() for s in line.split("=", 1)]
            if key in ("name", "version", "hash"):
                block[key] = value.strip('"')
    if block.get("name"):
        out.append(block)
    return [p for p in out if p["name"].startswith(("mlbend-", "bend-ml-", "bendtorch-"))]


def machine():
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


def manifest_for(task, cfg, out, build_seconds, times, repeat, probe, trainer="bend"):
    run_seconds = min(times) if times else 0.0
    start = time.perf_counter()
    reference_values = cfg["reference"]()
    reference_seconds = time.perf_counter() - start
    if trainer == "numpy":
        bend_values = reference_values
        table = compare(cfg, reference_values, reference_values)
    else:
        bend_values = cfg["parse"](out)
        table = compare(cfg, bend_values, reference_values)
    manifest = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": {"name": task, "description": cfg["description"], "program": cfg["program"],
                 "dataset": dict(cfg["dataset"], **probe), "seed": cfg.get("seed")},
        "trainer": trainer,
        "git": git_state(),
        "bend": {"version": bend_version()},
        "packages": packages(),
        "machine": machine(),
        "run": {
            "build_seconds": round(build_seconds, 3),
            "seconds_min": round(run_seconds, 3),
            "seconds_all": [round(t, 3) for t in times],
            "repeat": repeat,
            "fetch_seconds": probe["fetch_seconds"],
            "compute_seconds": round(max(0.0, run_seconds - probe["fetch_seconds"]), 3),
            "reference_seconds": round(reference_seconds, 3),
            "speed_ratio": round(reference_seconds / run_seconds, 3) if run_seconds else None,
        },
        "metrics": table,
        "passed": all(row["pass"] for row in table.values()),
    }
    return manifest


# --------------------------------------------------------------------------
# the report card and Weights & Biases
# --------------------------------------------------------------------------

def report_text(manifest):
    lines = [f"task: {manifest['task']['name']}  ({manifest['task']['description']})",
             f"git: {manifest['git']['branch']} {manifest['git']['commit']}"
             + (" (dirty)" if manifest["git"]["dirty"] else ""),
             f"trainer: {manifest['trainer']}   bend: {manifest['bend']['version']}   build: {manifest['run']['build_seconds']} s"
             f"   run: {manifest['run']['seconds_min']} s"
             f"   reference: {manifest['run']['reference_seconds']} s"
             f"   fetch: {manifest['run']['fetch_seconds']} s"
             f"   sha256: {manifest['task']['dataset']['sha256'][:12]}",
             "",
             f"{'metric':<18}{'bend':>14}{'reference':>14}{'delta':>12}  ok"]
    for name, row in manifest["metrics"].items():
        bend = "-" if row["bend"] is None else (f"{row['bend']:.6g}" if isinstance(row["bend"], float) else str(row["bend"]))
        want = "-" if row["reference"] is None else (f"{row['reference']:.6g}" if isinstance(row["reference"], float) else str(row["reference"]))
        delta = "-" if row.get("delta") is None else (f"{row['delta']:.4g}" if isinstance(row["delta"], float) else "-")
        lines.append(f"{name:<18}{bend:>14}{want:>14}{delta:>12}  {'ok' if row['pass'] else 'FAIL'}")
    lines.append("")
    lines.append("PASS" if manifest["passed"] else "FAIL")
    return "\n".join(lines)


def log_wandb(manifest, project, name, offline):
    try:
        import wandb
    except ImportError:
        raise SystemExit("wandb is not installed: reference/.venv/bin/python -m pip install wandb")

    mode = "offline" if offline or not os.environ.get("WANDB_API_KEY") else "online"
    run = wandb.init(project=project, name=name, config=manifest, mode=mode,
                     tags=[manifest["task"]["name"]], reinit=True)
    values = {}
    for metric, row in manifest["metrics"].items():
        if isinstance(row["bend"], (int, float)):
            values[metric] = row["bend"]
        if isinstance(row["reference"], (int, float)):
            values["reference/" + metric] = row["reference"]
    for metric, row in manifest["metrics"].items():
        if isinstance(row.get("delta"), (int, float)):
            values["delta/" + metric] = row["delta"]
    run.log(values)
    run.summary["passed"] = manifest["passed"]
    run.summary["git_commit"] = manifest["git"]["commit"]
    run.summary["bend_version"] = manifest["bend"]["version"]
    run.finish()
    return mode


def write_report(manifest, out_dir):
    """Write the timestamped report and the task's latest report."""
    os.makedirs(out_dir, exist_ok=True)
    latest_name = "latest" if manifest.get("trainer", "bend") == "bend" else "latest-numpy"
    latest_dir = os.path.join(os.path.dirname(out_dir), latest_name)
    os.makedirs(latest_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(out_dir, f"{manifest['task']['name']}-{stamp}.json")
    for target in (path, os.path.join(latest_dir, manifest["task"]["name"] + ".json")):
        with open(target, "w") as f:
            json.dump(manifest, f, indent=2)
    return path


def summary(root_dir):
    """Write RESULTS.md from the latest report of every task."""
    latest_dir = os.path.join(root_dir, "latest")
    if not os.path.isdir(latest_dir):
        raise SystemExit(f"no reports in {latest_dir}: run --all first")
    rows = [json.load(open(os.path.join(latest_dir, f)))
            for f in sorted(os.listdir(latest_dir)) if f.endswith(".json")]
    limit = len(rows[0]["metrics"]) if rows else 0
    lines = [
        "# Results",
        "",
        "The latest report card of every task, written by `reference/experiment.py`.",
        "`fetch` is the time to fill and hash the local cache; the Bend program reads",
        "the same bytes through `MLBEND_DATA`, so on a warm cache it is 0 and `compute`",
        "is the run itself. `Python` is the reference computation. The build time is",
        "Bend's compiler and is reported separately.",
        "",
        f"Machine: {rows[0]['machine']['platform']} · {rows[0]['machine']['cpu_count']} CPUs ·"
        f" Bend {rows[0]['bend']['version']}" if rows else "",
        "",
        "| task | dataset | hash | metric | Bend | reference | delta | fetch ms | Bend ms | compute ms | Python ms | ok |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|",
    ]
    for m in rows:
        for name, row in m["metrics"].items():
            b = row["bend"]
            r = row["reference"]
            delta = row.get("delta")
            ratio = m["run"]["speed_ratio"]
            fmt = lambda v: "-" if v is None else (f"{v:.6g}" if isinstance(v, float) else str(v))
            lines.append(
                f"| `{m['task']['name']}` | {m['task']['dataset']['name']} "
                f"({m['task']['dataset']['rows']}x{m['task']['dataset']['features']}) "
                f"| `{m['task']['dataset']['sha256'][:8]}` "
                f"| {name} | {fmt(b)} | {fmt(r)} | {fmt(delta)} "
                f"| {m['run']['fetch_seconds'] * 1000:.0f} "
                f"| {m['run']['seconds_min'] * 1000:.0f} "
                f"| {m['run']['compute_seconds'] * 1000:.0f} "
                f"| {m['run']['reference_seconds'] * 1000:.0f} "
                f"| {'ok' if row['pass'] else 'FAIL'} |")
    lines += ["", "The dataset, seed, hub hashes and the git state of every run are in",
              "`results/latest/<task>.json`; the timestamped reports are in `results/raw/`.", ""]
    path = os.path.join(root_dir, "RESULTS.md")
    open(path, "w").write("\n".join(lines))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASKS), help="the task to run")
    parser.add_argument("--all", action="store_true", help="every task")
    parser.add_argument("--list", action="store_true", help="say what the tasks are")
    parser.add_argument("--wandb", action="store_true", help="log the manifest to Weights & Biases")
    parser.add_argument("--offline", action="store_true", help="force W&B offline mode")
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "mlbend"))
    parser.add_argument("--out", default=os.path.join(ROOT, "results/raw"), help="where the reports go")
    parser.add_argument("--repeat", type=int, default=1, help="run each program this many times and keep the minimum")
    parser.add_argument("--trainer", choices=["bend", "numpy"], default="bend",
                        help="which trainer runs the task: the Bend program or the NumPy reference")
    parser.add_argument("--summary", action="store_true", help="write results/RESULTS.md from the latest reports")
    args = parser.parse_args()

    if args.summary:
        print(summary(os.path.join(ROOT, "results")))
        return 0

    if args.list:
        for name, cfg in TASKS.items():
            print(f"{name:<18} {cfg['program']:<32} {cfg['description']}")
        return 0

    names = sorted(TASKS) if args.all else ([args.task] if args.task else [])
    if not names:
        parser.error("give --task, --all or --list")

    failed = False
    for name in names:
        cfg = TASKS[name]
        probe = dataset_probe(cfg)
        if args.trainer == "numpy":
            print(f"{name}: the NumPy trainer (no Bend build)")
            out, build_seconds, times = "", 0.0, []
        else:
            out, build_seconds, times, _ = run_bend(name, cfg, args.repeat)
        manifest = manifest_for(name, cfg, out, build_seconds, times, args.repeat, probe, args.trainer)
        path = write_report(manifest, args.out)
        print(report_text(manifest))
        print(f"report: {path}")
        if args.wandb:
            mode = log_wandb(manifest, args.project, f"{name}-{manifest['git']['commit']}", args.offline)
            print(f"wandb: {mode}")
        print()
        failed = failed or not manifest["passed"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
