# ss25-nn-project

My final project for the seminar "Introduction to Neural Networks" of summer
semester 2025.

## Description

We were tasked to implement a small neural network from scratch in PyTorch,
based on the things we have learned in the seminar. I chose to implement a
simple cross-encoder for passage-based ranking (aka a reranker for reranking
in RAG systems).

Further more, I tried to experiment with different learning approaches in IR
ranking:
1. *pointwise*
2. *pairwise*
3. *listwise*

For orientation, here are the most important locations in the project:

```
.
│── README.md           # ← you are here
│── report/  
│   └── main.pdf        # final project report
│── experiments/        # Jupyter notebooks for training & evaluation
│   │── pointwise.ipynb # one notebook per learning approach
│   │── ...
│   └── listwise.ipynb
└── src/
    │── data/           # datasets
    │── loss/           # custom loss function(s)
    │── metrics/        # custom metrics
    └── models/         # the PyTorch model implementations (imported in the notebooks)
```

For more details, refer to the [project report](./report/main.pdf) and
[`docs.md`](./docs.md) for an overview and introduction to the code.

## Installation

First, clone this repository:

```bash
git clone https://github.com/lsg551/ss25-nn-project
cd ss25-nn-project
```

> [!TIP]
> To review the official submission (deadline 2025-09-30 23:59),
> checkout the `submission` ref (if the latest commit changed after the deadline):
> ```bash
> git fetch --all --tags
> git checkout submission
> ```

To install the required dependencies, you can either use the
[`uv` package manager](https://github.com/astral-sh/uv), or fall back to `pip`:

```bash
# with uv
uv install

# or with IPython support
uv install --group dev

# with pip (fallback)
pip install -r requirements.txt
```

> [!NOTE]
> Because `torch` and other packages are hardware-specific, you may run into
> issues and want to ignore the lockfile (for reproducibility) and install
> from `pyproject.toml` directly.

Also, run `make help` to print a list of available commands.

## Reproducing The Results

The key pieces are the notebooks in the `experiments/` folder. Each notebook
contains the code to train and evaluate a model with a specific learning approach.

After installing the dependencies, you can run the notebooks yourself.
Note that sufficient system memory and GPU VRAM is required to train the models.

## License

The code in this repository is licensed under the MIT license (see
[LICENSE](./LICENSE) for details). Paper and similiar documents as well as
datsets used for training and evaluation may be subject to different licenses.