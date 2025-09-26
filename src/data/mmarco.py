"""mMARCO dataset.

Use the `mMARCO` class directly or `mMARCO.as_dataloaders()` to create a
streaming dataset as PyTorch's 'DataLoader' to use in a PyTorch training
loop directly.
"""

import math
from itertools import chain
from typing import Optional, TypedDict, cast, override

import torch
from datasets import IterableDataset as HFIterableDataset
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset as TorchIterableDataset

from src.data.itersplit import IterSplit, Split


class mMARCOSample(TypedDict):
    """A single training sample from mMARCO."""

    query: str
    positive: str
    negative: str


class mMARCOBinaryTask(TypedDict):
    """A single binary classification task from mMARCO.

    This is how one actual sample is framed as two separate binary tasks
    in the training loop.
    """

    query: str
    candidate: str  # either positive or negative
    label: torch.Tensor  # shape (1,) with 1 or 0


def _sample_as_binary_task(sample: mMARCOSample) -> list[mMARCOBinaryTask]:
    """Format a single mMARCOSample as two binary tasks."""
    return [
        mMARCOBinaryTask(
            query=sample["query"],
            candidate=sample["positive"],
            label=torch.tensor(1.0, dtype=torch.float),
        ),
        mMARCOBinaryTask(
            query=sample["query"],
            candidate=sample["negative"],
            label=torch.tensor(0.0, dtype=torch.float),
        ),
    ]


class mMARCOBatch(TypedDict):
    """How a batch of mMARCO samples looks like after collate_fn.

    Returned from DataLoader"""

    queries: list[str]
    candidates: list[str]
    labels: torch.Tensor  # shape (2 * batch_size, 1)


# NOTE: the purpose of `collate_fn` in PyTorch's DataLoader is to customize the
# batching data structure. By default, DataLoader takes a subset from the dataset
# and combines them into a batch of tensors.
def collate_fn(samples: list[mMARCOSample]) -> mMARCOBatch:
    tasks = list(chain.from_iterable(_sample_as_binary_task(s) for s in samples))
    labels = torch.stack([t["label"] for t in tasks])  # shape (2*B,)
    return mMARCOBatch(
        queries=[t["query"] for t in tasks],
        candidates=[t["candidate"] for t in tasks],
        labels=labels,
    )


class mMARCO(TorchIterableDataset):
    """mMARCO dataset.

    mMARCO is a dataset introduced by XXX et al. (XXX) and is built on top of MS MARCO,

    Example:
    >>> train_dl, val_dl, test_dl = mMARCO.as_dataloaders()
    """

    def __init__(
        self,
        *,
        split: Optional[Split] = None,
        fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        shuffle: bool = True,
        shuffle_buffer_size: int = 10_000,
        max_samples: Optional[int] = None,
        unique_key: str = "query",
    ) -> None:
        """Create a new mMARCO streaming dataset instance.

        The dataset is not loaded into memory, but streamed from the source
        on demand.

        Args:
            split (Optional[Split], optional): If provided, yields only samples
                from that split. If `None`, yields all. Defaults to None.
            fractions (tuple[float, float, float], optional): Split sizes for
                train/val/test sets. Must sum to ~1.0. Defaults to (0.8, 0.1, 0.1).
            seed (int, optional): Seed for reproducibility. Defaults to 42.
            shuffle (bool, optional): If True and split == 'train', apply
                streaming shuffle (HF buffer shuffle). Defaults to True.
            shuffle_buffer_size (int, optional): Buffer size for HF shuffle
                (trade-off randomness vs memory). I.e., how many samples are
                kept in memory. Defaults to 10_000.
            max_samples (Optional[int], optional): Optional cap (after split
                filtering). Useful to bound an "epoch"; if None, length is
                undefined (no __len__). Defaults to None.
            unique_key (str, optional): Field name used as unique text for
                hashing (default 'query').. Defaults to "query".

        Raises:
            ValueError: If the Hugging Face dataset could not be loaded as a
                `IterableDataset`.
            ValueError: If fractions do not sum to ~1.0 (within tolerance).
        """
        super().__init__()
        if not math.isclose(sum(fractions), 1.0):
            raise ValueError(f"fractions must sum to 1.0, got {fractions}")
        self.split = split
        self.fractions = fractions
        self.seed = seed
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.max_samples = max_samples
        self.unique_key = unique_key
        self._assigner = IterSplit(fractions, seed=seed)

        # Load streaming base dataset (single 'train' split on hub)
        self._data: HFIterableDataset = cast(
            HFIterableDataset,
            load_dataset(
                "unicamp-dl/mmarco",
                data_dir="english",
                revision="refs/convert/parquet",
                split="train",
                streaming=True,
            ),
        )

    @override
    def __iter__(self):
        ds = self._data

        # shuffle only for trainset (common convention) and only if requested
        if self.shuffle and (self.split == "train" or self.split is None):
            # HF IterableDataset supports streaming shuffle with buffer
            ds = ds.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer_size)

        # worker sharding
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            ds = ds.shard(worker_info.num_workers, worker_info.id)

        # filtering for the selected split
        # NOTE: to read more about this, look at src/data/itersplit.py
        if self.split is not None:
            target_split = self.split
            assigner = self._assigner
            key_field = self.unique_key
            ds = ds.filter(lambda ex: assigner(ex[key_field]) == target_split)

        # Yield samples converting to the TypedDict struct
        for i, sample in enumerate(ds):
            if self.max_samples is not None and i >= self.max_samples:
                break
            # Expect keys 'query', 'positive', 'negative';
            # raise if missing to surface format mismatch early.
            try:
                yield mMARCOSample(
                    query=sample["query"],
                    positive=sample["positive"],
                    negative=sample["negative"],
                )
            except KeyError as e:
                missing = e.args[0]
                raise KeyError(
                    f"Missing expected key '{missing}' in sample; available keys: {list(sample.keys())}"
                ) from None

    # NOTE: No __len__ on purpose unless max_samples provided (could add conditional version if wanted)
    def __len__(self):  # pragma: no cover - only defined if bounded
        if self.max_samples is None:
            raise TypeError(
                "Length undefined for streaming dataset without max_samples. "
                "Pass max_samples to enable deterministic epoch length."
            )
        return self.max_samples

    @classmethod
    def as_dataloaders(
        cls,
        *,
        batch_size: int = 32,
        fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        shuffle_train: bool = True,
        shuffle_buffer_size: int = 10_000,
        max_samples_per_split: Optional[int] = None,
        num_workers: int = 0,  # 0 = no sub-processes
        collate_fn=collate_fn,
    ) -> tuple[
        DataLoader[mMARCOBatch], DataLoader[mMARCOBatch], DataLoader[mMARCOBatch]
    ]:
        """Get the mMARCO dataset as a PyTorch `DataLoader` for each split.

        Each split is its own dataset instance filtering the shared streaming
        source. If `max_samples_per_split` is set, each split yields at most
        that many *original* samples
        
        NOTE: Default `collate_fn` expands each into two binary tasks.
        Therefore, the argument `max_samples_per_split` is divided by two.

        Args:
            batch_size (int, optional): Samples aggregated in batches for
                efficiency. NOTE: because the default implementation of `collate_fn`
                effectively doubles the batch size, `batch_size` is divided by 2 for
                user-defined batch sizes. Defaults to 32.
            fractions (tuple[float, float, float], optional): Split configuration.
                Defaults to (0.8, 0.1, 0.1).
            seed (int, optional): Seed for reproducibility. Defaults to 42.
            shuffle_train (bool, optional): Whether to shuffle the train split.
                Defaults to True.
            shuffle_buffer_size (int, optional): Buffer size for HF shuffle
                (trade-off randomness vs memory). Defaults to 10_000.
            max_samples (Optional[int], optional): Optional cap (after split
                filtering). Useful to bound an "epoch"; if None, length is
                undefined (no __len__). Defaults to None.
            num_workers (int, optional): How many additional subprocesses are
                spawned to load data in parallel. Defaults to 0.

        Returns:
            tuple[ DataLoader[mMARCOBatch], DataLoader[mMARCOBatch], DataLoader[mMARCOBatch] ]:
                a 3-tuple of DataLoaders for train, val, test splits.
        """

        if max_samples_per_split is not None:
            # HACK: account for collate_fn doubling the number of samples
            max_samples_per_split = max_samples_per_split // 2

        # HACK: account for collate_fn doubling the number of samples
        batch_size = batch_size // 2
        if batch_size < 1:
            raise ValueError(
                "batch_size too small; must be at least 2 to account for collate_fn doubling samples"
            )
    

        # NOTE: the `DataLoader` signature usually does not permit an `IterableDataset`,
        # but in the docstring: … risky, but works (probably) …

        train_ds = cls(
            split="train",
            fractions=fractions,
            seed=seed,
            shuffle=shuffle_train,
            shuffle_buffer_size=shuffle_buffer_size,
            max_samples=max_samples_per_split,
        )
        val_ds = cls(
            split="validation",
            fractions=fractions,
            seed=seed,
            shuffle=False,
            shuffle_buffer_size=shuffle_buffer_size,
            max_samples=max_samples_per_split,
        )
        test_ds = cls(
            split="test",
            fractions=fractions,
            seed=seed,
            shuffle=False,
            shuffle_buffer_size=shuffle_buffer_size,
            max_samples=max_samples_per_split,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            # TODO: for even faster processing, enable these
            # But read the docs about `pin_memory` first, there's something to
            # consider if collate_fn is custom
            # pin_memory=True,
            # persistent_workers=True
        )

        return train_loader, val_loader, test_loader
