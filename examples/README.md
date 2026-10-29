# The examples

One directory per task, in the four areas of the corpus. Each task directory
holds a `main.bend`, a `README.md` with its command and its measured numbers,
and a `plot.svg`.

| task | what it shows |
|---|---|
| [`classification/knn`](classification/knn) | three nearest neighbours over breast cancer: no gradients, no library, a distance and a vote |
| [`classification/logistic`](classification/logistic) | logistic regression over iris with bend-ml's tensors and autodiff, step by step |
| [`classification/mlp`](classification/mlp) | a two-layer rectified network over iris: the backward chain through both layers |
| [`classification/naive_bayes`](classification/naive_bayes) | the Gaussian baseline over breast cancer: the gradient-free reference point |
| [`classification/naive_bayes_heart`](classification/naive_bayes_heart) | the same baseline on heart failure: a second dataset |
| [`classification/crossval`](classification/crossval) | the whole chain: a scaler fitted on each training fold, logistic regression, stratified folds, no leakage |
| [`diffusion/2d`](diffusion/2d) | a DDPM denoiser trained and sampled on two moons |
| [`diffusion/digits`](diffusion/digits) | the same over 1024 real 8×8 optdigits images |
| [`diffusion/vit`](diffusion/vit) | a transformer denoiser on the same digits, against a PyTorch twin and the lucidrains UNet |
| [`transformers/attention`](transformers/attention) | a single-head self-attention block in plain Bend, checked against NumPy |
| [`transformers/periodic`](transformers/periodic) | a one-block causal transformer trained on a periodic sequence |
| [`unsupervised/kmeans`](unsupervised/kmeans) | k-means over wine, through `mlbend-cluster` |
| [`unsupervised/pca`](unsupervised/pca) | the first principal component of wine by power iteration |

These are programs that use the libraries and the packages, not packages of
their own. Bend-ml and BendTorch are the libraries; nothing here re-implements
them.

Every task README opens with the command that runs it:

```console
$ bendenv run -- bend examples/diffusion/vit/main.bend
```

Most of them are also tasks in the experiment layer, where the same number is
computed again with NumPy or scikit-learn:

```console
$ reference/.venv/bin/python reference/experiment.py --list
```
