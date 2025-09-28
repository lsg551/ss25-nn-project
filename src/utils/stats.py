from dataclasses import dataclass, field
from datetime import datetime

import torch

type Losses = list[list[list[float]]]  # (epoch, batch, sample)


@dataclass
class Stats:
    """Class to keep track of training statistics."""

    start: datetime = field(default_factory=datetime.now)
    end: datetime | None = None
    elapsed: float | None = None  # in seconds

    # keep track of each loss per sample (!) to accurately calculate and plot std & CV
    train_losses: Losses = field(default_factory=list)
    val_losses: Losses = field(default_factory=list)
    test_losses: Losses = field(default_factory=list)

    def collect(
        self, split: str, epoch: int, batch: int, sample_loss: float | list[float]
    ):
        """Collect a per-sample loss for a given split/epoch/batch.

        Dynamically grows the internal (epoch, batch, sample) structure
        as needed for convenience.
        """
        losses = self._losses(split)
        self._ensure_capacity(losses, epoch, batch)
        losses[epoch][batch].extend(
            sample_loss if isinstance(sample_loss, list) else [sample_loss]
        )

    def _losses(self, split: str) -> Losses:
        match split:
            case "train":
                return self.train_losses
            case "val":
                return self.val_losses
            case "test":
                return self.test_losses
            case _:
                raise ValueError(
                    f"Invalid split: {split}. Must be 'train', 'val', or 'test'."
                )

    # TODO: improve perf
    @staticmethod
    def _ensure_capacity(losses: Losses, epoch: int, batch: int) -> None:
        # Ensure epochs list is long enough
        while len(losses) <= epoch:
            losses.append([])
        # Ensure batches list for this epoch is long enough
        while len(losses[epoch]) <= batch:
            losses[epoch].append([])

    @staticmethod
    def _flatten_epoch(losses: Losses, epoch: int) -> list[float]:
        """Flatten all sample losses for a given epoch into a single list (=reduce batches)."""
        if epoch < 0 or epoch >= len(losses):
            return []
        # Concatenate batches in order for the given epoch
        out: list[float] = []
        for batch_losses in losses[epoch]:
            out.extend(batch_losses)
        return out

    def mean(self, split: str, epoch: int, window: int = 0) -> float:
        """Calculate the mean loss over the specified window of samples in an epoch.

        Args:
            split (str): "train", "val", or "test"
            epoch (int): Epoch number
            window (int, optional): Last n samples. Useful to get the rolling mean.
                If 0, use all samples in the epoch. Defaults to 0.

        Returns:
            float: Mean loss
        """
        losses = self._losses(split)
        samples = self._flatten_epoch(losses, epoch)
        if not samples:
            return float("nan")
        if window and window > 0:
            samples = samples[-window:]
        t = torch.tensor(samples, dtype=torch.float32)
        return float(t.mean().item())

    def std(self, split: str, epoch: int, window: int = 100) -> float:
        """Calculate the standard deviation of the loss over the specified window of samples in an epoch.

        Args:
            split (str): "train", "val", or "test"
            epoch (int): Epoch number
            window (int, optional): Last n samples. Useful to get the rolling std.
                If 0, use all samples in the epoch. Defaults to 100.

        Returns:
            float: Standard deviation of the loss
        """
        losses = self._losses(split)
        samples = self._flatten_epoch(losses, epoch)
        if not samples:
            return float("nan")
        if window and window > 0:
            samples = samples[-window:]
        t = torch.tensor(samples, dtype=torch.float32)
        # population standard deviation (unbiased=False)
        return float(t.std(unbiased=False).item())

    def CV(
        self,
        split: str,
        epoch: int,
        window: int = 100,
    ) -> float:
        """Calculate the coefficient of variation (CV) of the loss over the specified window of samples in an epoch.

        Args:
            split (str): "train", "val", or "test"
            epoch (int): Epoch number
            window (int, optional): Last n samples. Useful to get the rolling CV.
                If 0, use all samples in the epoch. Defaults to 100.

        Returns:
            float: Coefficient of variation of the loss
        """
        mu = self.mean(split, epoch, window if window else 0)
        sigma = self.std(split, epoch, window if window else 0)
        if mu == 0.0 or not (mu == mu) or not (sigma == sigma):  # handle NaN
            return float("nan")
        return float(sigma / mu)

    def stop(self):
        """Stop the timer and compute elapsed time in seconds."""
        self.end = datetime.now()
        self.elapsed = (self.end - self.start).total_seconds()
        return self.elapsed
