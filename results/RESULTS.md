# Results

The latest report card of every task, written by `reference/experiment.py`.
The Bend time is the compiled program's run (minimum of the repeats); the
Python time is the reference computation. Both include the dataset fetch.
The build time is Bend's compiler and is reported separately.

Machine: Linux-7.2.9-200.fc44.x86_64-x86_64-with-glibc2.43 · 28 CPUs · Bend 2.0.37

| task | dataset | metric | Bend | reference | delta | Bend ms | Python ms | speed (Python/Bend) | ok |
|---|---|---|---:|---:|---:|---:|---:|---:|:--:|
| `cancer-knn` | breast_cancer (683x9) | accuracy | 0.978102 | 0.978102 | 1.0219e-08 | 857 | 1450 | 1.69x | ok |
| `cancer-naive-bayes` | breast_cancer (683x9) | accuracy | 0.985401 | 0.985401 | -9.85401e-09 | 701 | 680 | 0.97x | ok |
| `iris-crossval` | iris (150x4) | mean_accuracy | 0.933333 | 0.953333 | -0.02 | 701 | 1345 | 1.92x | ok |
| `iris-crossval` | iris (150x4) | fold_min | 0.833333 | 0.9 | - | 701 | 1345 | 1.92x | ok |
| `iris-crossval` | iris (150x4) | fold_max | 0.966667 | 1 | - | 701 | 1345 | 1.92x | ok |
| `iris-mlp` | iris (150x4) | accuracy | 0.973333 | 0.973333 | 2.66667e-08 | 1098 | 252 | 0.23x | ok |
| `wine-kmeans` | wine (6497x11) | inertia | 50645.6 | 50645.6 | -0.00456047 | 4144 | 1097 | 0.27x | ok |
| `wine-kmeans` | wine (6497x11) | cluster_sizes | [2384, 990, 3123] | [2384, 990, 3123] | 0 | 4144 | 1097 | 0.27x | ok |
| `wine-pca` | wine (6497x11) | explained_variance_ratio | 0.275443 | 0.275443 | 3.95586e-07 | 6112 | 1147 | 0.19x | ok |
| `wine-regression` | wine (6497x11) | mse_after | 0.707852 | 0.707863 | -1.1515e-05 | 28298 | 1430 | 0.05x | ok |

The dataset, seed, hub hashes and the git state of every run are in
`results/latest/<task>.json`; the timestamped reports are in `results/raw/`.
