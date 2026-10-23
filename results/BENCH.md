# Kernel benchmarks

Compute only: no dataset, no IO. `make bench` writes this file.

Machine: Linux-7.2.9-200.fc44.x86_64-x86_64-with-glibc2.43 · 28 CPUs · NumPy 2.5.4

| kernel | shape x rounds | Bend | NumPy | Bend GFLOP/s | NumPy GFLOP/s | speed (NumPy/Bend) |
|---|---|---:|---:|---:|---:|---:|
| `matmul` | 128x128 x50 | 20.9 ms | 1.0 ms | 10.0 | 201.1 | 0.05x |

NumPy uses its BLAS and every thread it likes; the Bend product here is
sequential (`par = 0`). The comparison is one thread of Bend against
whatever NumPy chooses, which is the honest reading of the two
programs as they are run.
