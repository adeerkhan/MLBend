"""The dataset cache: MLBEND_DATA points the loader at a local file.

  reference/.venv/bin/python reference/test_datasets_cache.py

Writes a five-row iris file under `mstz/iris/iris.data` in a temporary
directory, points `MLBEND_DATA` at it, and runs `datasets/tests/iris.bend`.
The program must report five rows, not the 150 it would fetch.
"""
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
if os.path.isdir(ENV_DIR):
    ENV["BEND_LIB"] = ENV_DIR

ROWS = [
    "5.1,3.5,1.4,0.2,Iris-setosa",
    "4.9,3.0,1.4,0.2,Iris-setosa",
    "7.0,3.2,4.7,1.4,Iris-versicolor",
    "6.4,3.2,4.5,1.5,Iris-versicolor",
    "6.3,3.3,6.0,2.5,Iris-virginica",
]


def main():
    cache = tempfile.mkdtemp()
    path = os.path.join(cache, "mstz", "iris", "iris.data")
    os.makedirs(os.path.dirname(path))
    open(path, "w").write("\n".join(ROWS) + "\n")
    env = dict(ENV, MLBEND_DATA=cache)
    binary = os.path.join(tempfile.mkdtemp(), "mlbend-datasets-cache")
    build = subprocess.run([BEND, "-o", binary, "datasets/tests/iris.bend"],
                           capture_output=True, text=True, env=env, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit("the cache test did not build:\n" + build.stdout + build.stderr)
    out = subprocess.run([binary], capture_output=True, text=True, env=env, cwd=ROOT)
    if out.returncode != 0:
        raise SystemExit("the cache test did not run:\n" + out.stdout + out.stderr)
    match = re.search(r"rows: (\d+)", out.stdout)
    got = int(match.group(1)) if match else -1
    print(out.stdout.strip())
    if got != len(ROWS):
        print(f"ERROR the cache was not read: {got} rows, expected {len(ROWS)}")
        return 1
    print(f"ok   the cache was read: {got} rows")
    # the cache is authoritative: an empty directory must fail, not fetch
    empty = tempfile.mkdtemp()
    miss = subprocess.run([binary], capture_output=True, text=True,
                          env=dict(env, MLBEND_DATA=empty), cwd=ROOT)
    if "the dataset failed" not in miss.stdout:
        print("ERROR a cache miss did not fail; it may have fetched instead")
        print(miss.stdout + miss.stderr)
        return 1
    print("ok   a cache miss fails instead of fetching")
    return 0


if __name__ == "__main__":
    sys.exit(main())
