"""MS MARCO dataset.

https://huggingface.co/datasets/microsoft/ms_marco
"""

from typing import Generator, Literal, TypedDict, cast

from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset as TorchIterableDataset


class MSMARCOPassages(TypedDict):
    """List of candidate passages retrieved for a given query."""

    is_selected: list[Literal[-1, 0, 1]]
    """Annotation indicating whether the passage is relevant (`1`), irrelevant
    (`0`) or not judged (`-1`).
    
    In validation/test set, this will often be `-1` to avoid data leakage.
    Either ignore these or treat them as non-relevant (although this is not
    ideal as it introduces bias).

    It's not guaranteed that there is exactly one relevant passage per query,
    it could be none at all (all `0` or `-1`).
    """
    passage_text: list[str]
    """The text of the passage."""
    url: list[str]
    """List of URLs corresponding to the passages."""


class MSMARCOSample(TypedDict):
    """Single sample from the MS MARCO dataset."""

    query: str
    """Natural language question asked by a user."""
    query_id: int
    """"Unique identifier for the query."""
    query_type: str
    """Type of the question, e.g. `DESCRIPTION` = asking for an explanation."""
    answers: list[str]
    """List of free-form answers to the query.
    
    Usually one, sometime more.

    NOTE: There might be no answers at all in validation/test sets to avoid
    data leakage.
    """
    passages: MSMARCOPassages
    """List of candidate passages."""
    well_formed_answers: list[str]
    """Smaller subset of high-quality answers. Usually empty."""


class MSMARCOBatch(TypedDict):
    """Batch of samples from the MS MARCO dataset used for training.

    This is the data structure after collating multiple `MSMARCOSample`
    entries. Each batch is supposed to be a training/val/test batch
    for PyTorch.

    Some information is omitted or restructured for simplicity.

    Example:
    >>> for idx in range(len(batch["queries"])):
    ...    query = batch["queries"][idx]
    ...    passages = batch["passages"][idx]
    ...    labels = batch["labels"][idx]
    """

    queries: list[str]
    """List of queries in the batch."""
    candidates: list[list[str]]
    """List of candidate passages for each query in the batch."""
    labels: list[list[Literal[-1, 0, 1]]]
    """List of relevance labels for each candidate passage in the batch."""


def collate_fn(batch: list[MSMARCOSample]) -> MSMARCOBatch:
    """Custom PyTorch collate function to batch MSMARCO samples.

    Takes a list of `MSMARCOSample` items (the batch) and restructures them
    into a single `MSMARCOBatch` dictionary.
    """
    return {
        "queries": [item["query"] for item in batch],
        "candidates": [item["passages"]["passage_text"] for item in batch],
        "labels": [item["passages"]["is_selected"] for item in batch],
    }


class MSMARCO(TorchIterableDataset):
    """MS MARCO dataset.

    MS MARCO (Microsoft MAchine Reading COmprehension) is a large-scale dataset for
    training and evaluating machine reading comprehension models, including IR
    ranking tasks. It consists of real-world questions and answers, along with
    relevant passages from web documents.

    It was introduced at NIPS 2016 by Bajaj et al. https://arxiv.org/pdf/1611.09268.

    NOTE: This wrapper uses revision `v2.1`!
    """

    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self._data = cast(
            DatasetDict,
            load_dataset(
                "microsoft/ms_marco",
                "v2.1",
                # streaming=True,
            ),
        )

    def __iter__(self) -> Generator[MSMARCOBatch, None, None]:
        """Get a lazy iterator over the training samples."""
        split = self._data["train"]
        return split.iter(self.batch_size)  # pyright: ignore[reportReturnType]

    @staticmethod
    def as_dataloaders(
        batch_size: int = 32,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Get train, validation and test dataloaders.

        Returns:
            A tuple of three `DataLoader` objects for training, validation, and testing.
        """

        # NOTE: the `DataLoader` signature usually does not permit an `IterableDataset`,
        # but in the docstring: … risky, but works (probably) …

        dataset = MSMARCO()
        train_loader = DataLoader(
            dataset._data["train"],  # pyright: ignore[reportArgumentType]
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=True,
        )
        val_loader = DataLoader(
            dataset._data["validation"],  # pyright: ignore[reportArgumentType]
            batch_size=batch_size,
            collate_fn=collate_fn,
        )
        test_loader = DataLoader(
            dataset._data["test"],  # pyright: ignore[reportArgumentType]
            batch_size=batch_size,
            collate_fn=collate_fn,
        )
        return train_loader, val_loader, test_loader
