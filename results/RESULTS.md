# Results

The latest report card of every task, written by `reference/experiment.py`.
`fetch` is the time to fill and hash the local cache; the Bend program reads
the same bytes through `MLBEND_DATA`, so on a warm cache it is 0 and `compute`
is the run itself. `Python` is the reference computation. The build time is
Bend's compiler and is reported separately.

Machine: Linux-7.2.9-200.fc44.x86_64-x86_64-with-glibc2.43 · 28 CPUs · Bend 2.0.38

| task | dataset | hash | metric | Bend | reference | delta | fetch ms | Bend ms | compute ms | Python ms | ok |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| `attention-parity` | synthetic-attention (4x8) | `439ef1cd` | checksum | 1.76295 | 1.76295 | 3.34167e-09 | 0 | 1 | 1 | 7 | ok |
| `cancer-knn` | breast_cancer (683x9) | `4c35090c` | accuracy | 0.978102 | 0.978102 | 1.0219e-08 | 0 | 225 | 225 | 1696 | ok |
| `cancer-naive-bayes` | breast_cancer (683x9) | `4c35090c` | accuracy | 0.985401 | 0.985401 | -9.85401e-09 | 0 | 31 | 31 | 999 | ok |
| `heart-naive-bayes` | heart_failure (299x12) | `e0771c8c` | accuracy | 0.933333 | 0.95 | -0.0166667 | 1 | 12 | 11 | 738 | ok |
| `iris-crossval` | iris (150x4) | `6f608b71` | mean_accuracy | 0.933333 | 0.953333 | -0.02 | 1 | 99 | 98 | 704 | ok |
| `iris-crossval` | iris (150x4) | `6f608b71` | fold_min | 0.833333 | 0.9 | - | 1 | 99 | 98 | 704 | ok |
| `iris-crossval` | iris (150x4) | `6f608b71` | fold_max | 0.966667 | 1 | - | 1 | 99 | 98 | 704 | ok |
| `iris-mlp` | iris (150x4) | `6f608b71` | accuracy | 0.973333 | 0.973333 | 2.66667e-08 | 0 | 574 | 574 | 339 | ok |
| `transformer-periodic` | periodic-0-7 (32x8) | `3fd593d8` | first_loss | 2.35914 | 2.35914 | 6.81277e-07 | 0 | 32 | 32 | 50 | ok |
| `transformer-periodic` | periodic-0-7 (32x8) | `3fd593d8` | final_loss | 0.00211434 | 0.1351 | - | 0 | 32 | 32 | 50 | ok |
| `transformer-periodic` | periodic-0-7 (32x8) | `3fd593d8` | accuracy | 1 | 0.9375 | 0.0625 | 0 | 32 | 32 | 50 | ok |
| `wine-kmeans` | wine (6497x11) | `7dcde7c7` | inertia | 50645.6 | 50645.6 | -0.00456047 | 1 | 2830 | 2829 | 1314 | ok |
| `wine-kmeans` | wine (6497x11) | `7dcde7c7` | cluster_sizes | [2384, 990, 3123] | [2384, 990, 3123] | 0 | 1 | 2830 | 2829 | 1314 | ok |
| `wine-pca` | wine (6497x11) | `7dcde7c7` | explained_variance_ratio | 0.275443 | 0.275443 | 3.95586e-07 | 0 | 5108 | 5108 | 1095 | ok |

The dataset, seed, hub hashes and the git state of every run are in
`results/latest/<task>.json`; the timestamped reports are in `results/raw/`.
