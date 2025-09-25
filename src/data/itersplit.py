"""Deterministic pseudo-splitting for streaming / iterable datasets.

Problem
=======

HuggingFace and PyTorch both support streaming of datasets via iterable
variants of their dataset classes. This allows to get samples or batches
on demand without loading the entire dataset into memory. Unfortunately,
splitting such datasets has never been implemented. One would have to load
the entire first, then split it.

Another, but slightly more complex method is to assign each sample to a split
on-the-fly. This raises concerns about reproducibility though.

Goal
====

We want to take a continuous (possibly endless or of unknown length) stream of
samples and assign each sample deterministically to exactly one of three logical
splits: train, validation, test. We can't pre-shuffle or pre-count. So we need a
per-sample decision procedure that's:
- deterministic (reproducible)
- stateless (no memory of previous samples, no dataset materialization)
- _approximately_ proportional to the desired fractions of the splits
- independent across worker processes (for PyTorch's `DataLoader`)

Idea
====

Cryptographic hash functions like MD5 are designed to spread input entropy
uniformly across their output bits, so non-adversarial inputs do tend to produce
outputs that appear uniformly random.

If we hash our samples and normalize to [0,1), this yields a uniform distribution
over this interval. We can then partition [0,1) into consecutive, non-overlapping
sub-intervals whose lengths are equal the target split fractions we specify.
Each sample is then assigned to the split whose sub-interval contains the hash.

NOTE: "adversarial" inputs would be inputs that are specifically designed to
produce MD5 hash collisions, which datasets unlikely do intentionally. Rare
exceptions could be UIDs, timestamps or randomly generated text.

Solution
========

First, a split must be fined, something like `(0.8, 0.1, 0.1)` for train,
validation, and test. Then, for each sample:

1. Take a _unique_ part of each sample
2. A seed is concatenated to this unique part as a string
3. The concatenated string is hashed (MD5) to a 32-bit integer
4. This integer is normalized to a float in [0,1)

The seed is applied to all hashes, this not only makes the split reproducible,
but allows to re-shuffle by changing the seed.

The assignment logic takes this hash (`value`) and partitions the interval
[0,1) into consecutive, non-overlapping sub-intervals whose lengths equal
the target split fractions. The sample is assigned to the split whose
sub-interval contains the hash.

In pseudo-code:

```
if value < train_end:
    split = 'train'
elif value < val_end:
    split = 'validation'
else:
    split = 'test'
```

Where `train_end` and `val_end` are just the partition boundaries.

Caveats
=======

- The supposedly _unique_ (part of the) sample may not be unique...
- No exact split sizes, but this should be good enough for large datasets.
- Batch processing not supported.
- No idea if there are any reasoning or implementation flaws in this approach.
  At best, this is measured empirically in the training/validation/test loops.
- I now this idea is used in practice for such things, but could not find a
  working example for PyTorch.
"""

import math
from hashlib import md5
from typing import Literal

type Split = Literal["train", "validation", "test"]


def hash(value: str, seed: int = 42) -> bytes:
    return md5((value + str(seed)).encode()).digest()


def normalize(hash: bytes) -> float:
    return int.from_bytes(hash[:4], "big") / 2**32  # value in [0,1)


def assign(hash: float, fractions: tuple[float, float, float]) -> Split:
    train_end = fractions[0]
    val_end = fractions[0] + fractions[1]
    # implicit: test split covers the rest [val_end, 1.0)

    if hash < train_end:
        return "train"
    elif hash < val_end:
        return "validation"
    else:
        return "test"


class IterSplit:
    """IterSplit assigns samples to splits based on a hash of a unique part.

    This is just a convenience wrapper around the functions `hash`, `normalize`
    and `assign`.

    Example:
    >>> assign = IterSplit((0.8, 0.1, 0.1), seed=42)
    >>> split = assign("some unique part of the sample")
    >>> print(split)  # "train", "validation" or "test"
    """

    def __init__(
        self, fractions: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 42
    ):
        if not math.isclose(sum(fractions), 1.0):
            raise ValueError(f"Split fractions must sum to 1.0, got {sum(fractions)}")

        self.fractions = fractions
        self.seed = seed

    def __call__(self, value) -> Split:
        h = hash(value, self.seed)
        u = normalize(h)
        return assign(u, self.fractions)
