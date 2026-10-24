# Results

The latest report card of every task, written by `reference/experiment.py`.
`fetch` is a Python probe of the dataset files; Bend fetches them inside its
own run, so `compute` is `Bend time - fetch` and is an estimate, not a kernel
measurement. `Python` is the reference computation. The build time is Bend's
compiler and is reported separately.

Machine: Linux-7.2.9-200.fc44.x86_64-x86_64-with-glibc2.43 · 28 CPUs · Bend 2.0.38

| task | dataset | hash | metric | Bend | reference | delta | fetch ms | Bend ms | compute ms | Python ms | ok |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| `attention-parity` | synthetic-attention (4x8) | `439ef1cd` | checksum | 1.76295 | 1.76295 | 3.34167e-09 | 0 | 1 | 1 | 6 | ok |
| `cancer-knn` | breast_cancer (683x9) | `4c35090c` | accuracy | 0.978102 | 0.978102 | 1.0219e-08 | 932 | 1015 | 83 | 1263 | ok |
| `cancer-naive-bayes` | breast_cancer (683x9) | `4c35090c` | accuracy | 0.985401 | 0.985401 | -9.85401e-09 | 779 | 902 | 123 | 910 | ok |
| `heart-naive-bayes` | heart_failure (299x12) | `e0771c8c` | accuracy | 0.933333 | 0.95 | -0.0166667 | 684 | 897 | 213 | 1652 | ok |
| `iris-crossval` | iris (150x4) | `6f608b71` | mean_accuracy | 0.933333 | 0.953333 | -0.02 | 727 | 847 | 120 | 848 | ok |
| `iris-crossval` | iris (150x4) | `6f608b71` | fold_min | 0.833333 | 0.9 | - | 727 | 847 | 120 | 848 | ok |
| `iris-crossval` | iris (150x4) | `6f608b71` | fold_max | 0.966667 | 1 | - | 727 | 847 | 120 | 848 | ok |
| `iris-mlp` | iris (150x4) | `6f608b71` | accuracy | 0.973333 | 0.973333 | 2.66667e-08 | 887 | 1080 | 193 | 170 | ok |
| `transformer-periodic` | periodic-0-7 (32x8) | `3fd593d8` | first_loss | 2.35914 | 2.35914 | 6.81277e-07 | 0 | 23 | 23 | 47 | ok |
| `transformer-periodic` | periodic-0-7 (32x8) | `3fd593d8` | final_loss | 0.00211434 | 0.1351 | - | 0 | 23 | 23 | 47 | ok |
| `transformer-periodic` | periodic-0-7 (32x8) | `3fd593d8` | accuracy | 1 | 0.9375 | 0.0625 | 0 | 23 | 23 | 47 | ok |
| `wine-kmeans` | wine (6497x11) | `7dcde7c7` | inertia | 50645.6 | 50645.6 | -0.00456047 | 1273 | 2993 | 1720 | 1241 | ok |
| `wine-kmeans` | wine (6497x11) | `7dcde7c7` | cluster_sizes | [2384, 990, 3123] | [2384, 990, 3123] | 0 | 1273 | 2993 | 1720 | 1241 | ok |
| `wine-pca` | wine (6497x11) | `7dcde7c7` | explained_variance_ratio | 0.275443 | 0.275443 | 3.95586e-07 | 1569 | 4788 | 3219 | 1331 | ok |

The dataset, seed, hub hashes and the git state of every run are in
`results/latest/<task>.json`; the timestamped reports are in `results/raw/`.
