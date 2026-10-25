# Results

The latest report card of every task, written by `reference/experiment.py`.
`fetch` is the time to fill and hash the local cache; the Bend program reads
the same bytes through `MLBEND_DATA`, so on a warm cache it is 0 and `compute`
is the run itself. `Python` is the reference computation. The build time is
Bend's compiler and is reported separately.

Machine: Linux-7.2.9-200.fc44.x86_64-x86_64-with-glibc2.43 · 28 CPUs · Bend 2.0.38

| task | dataset | hash | metric | seeds | Bend | reference | delta | fetch ms | Bend ms | compute ms | Python ms | ok |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| `attention-parity` | synthetic-attention (4x8) | `439ef1cd` | checksum | 1 | 1.76295 | 1.76295 | 3.342e-09 | 0 | 1 | 1 | 3 | ok |
| `cancer-knn` | breast_cancer (683x9) | `4c35090c` | accuracy | 1 | 0.978102 | 0.978102 | 1.022e-08 | 1 | 196 | 195 | 1617 | ok |
| `cancer-naive-bayes` | breast_cancer (683x9) | `4c35090c` | accuracy | 1 | 0.985401 | 0.985401 | -9.854e-09 | 0 | 22 | 22 | 535 | ok |
| `diffusion-2d` | two-moons (128x2) | `6764e1b1` | first_loss | 1 | 152.954 | 152.954 | -1.483e-05 | 0 | 30848 | 30848 | 35959 | ok |
| `diffusion-2d` | two-moons (128x2) | `6764e1b1` | final_loss | 1 | 0.0649613 | 0.0687128 | - | 0 | 30848 | 30848 | 35959 | ok |
| `diffusion-2d` | two-moons (128x2) | `6764e1b1` | sample_mean | 1 | 0.341456 | 0.4594 | - | 0 | 30848 | 30848 | 35959 | ok |
| `diffusion-2d` | two-moons (128x2) | `6764e1b1` | sample_std | 1 | 0.362767 | 0.630041 | - | 0 | 30848 | 30848 | 35959 | ok |
| `heart-naive-bayes` | heart_failure (299x12) | `e0771c8c` | accuracy | 1 | 0.95 | 0.95 | 0 | 1 | 9 | 8 | 769 | ok |
| `iris-crossval` | iris (150x4) | `6f608b71` | mean_accuracy | 5 | 0.930667 ± 0.006 | 0.930667 ± 0.006 | -5.867e-08 ± 1.4e-08 | 0 | 76 | 76 | 763 | ok |
| `iris-crossval` | iris (150x4) | `6f608b71` | fold_min | 5 | 0.84 ± 0.037 | 0.84 ± 0.037 | -2e-08 ± 1.8e-08 | 0 | 76 | 76 | 763 | ok |
| `iris-crossval` | iris (150x4) | `6f608b71` | fold_max | 5 | 0.986667 ± 0.018 | 0.986667 ± 0.018 | -1.067e-08 ± 1.5e-08 | 0 | 76 | 76 | 763 | ok |
| `iris-mlp` | iris (150x4) | `6f608b71` | accuracy | 1 | 0.973333 | 0.973333 | 2.667e-08 | 0 | 482 | 482 | 271 | ok |
| `transformer-periodic` | periodic-0-7 (32x8) | `3fd593d8` | first_loss | 1 | 2.35914 | 2.35914 | 6.813e-07 | 0 | 26 | 26 | 41 | ok |
| `transformer-periodic` | periodic-0-7 (32x8) | `3fd593d8` | final_loss | 1 | 0.00211434 | 0.1351 | - | 0 | 26 | 26 | 41 | ok |
| `transformer-periodic` | periodic-0-7 (32x8) | `3fd593d8` | accuracy | 1 | 1 | 0.9375 | 0.0625 | 0 | 26 | 26 | 41 | ok |
| `wine-kmeans` | wine (6497x11) | `7dcde7c7` | inertia | 1 | 50645.6 | 50645.6 | -0.00456 | 1 | 2289 | 2288 | 1073 | ok |
| `wine-kmeans` | wine (6497x11) | `7dcde7c7` | cluster_sizes | 1 | [2384, 990, 3123] | [2384, 990, 3123] | 0 | 1 | 2289 | 2288 | 1073 | ok |
| `wine-pca` | wine (6497x11) | `7dcde7c7` | explained_variance_ratio | 1 | 0.275443 | 0.275443 | 3.956e-07 | 0 | 4016 | 4016 | 1170 | ok |

Bend and reference are mean ± sample deviation over the seeds when a task
has more than one; a single-seed task has no deviation. The dataset, seeds,
hub hashes and the git state of every run are in
`results/latest/<task>.json`; the timestamped reports are in `results/raw/`.
