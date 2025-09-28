"""Convenience wrappers for common widget patterns in Jupyter Notebooks."""

from string import Formatter
from typing import Any, Callable

import ipywidgets
from IPython.display import display


class Label:
    """Convenience wrapper for an ipywidgets.Label that supports dynamic updates."""

    def __init__(
        self,
        name: str = "Label: {value}",
        *,
        show: bool = True,
        transform: Callable[[str | int | float], str | int | float] | None = None,
    ) -> None:
        self.name = name
        self.tmpl = Formatter().parse(name)

        placerholders = [field_name for _, field_name, _, _ in self.tmpl if field_name]
        if "value" not in placerholders:
            raise ValueError(
                "The format string 'name' must contain a 'value' placeholder."
            )

        self.widget = ipywidgets.Label(value=self.name.format(value=0))
        self.transform = transform

        if show:
            display(self.widget)

    def _render(self, value: Any):
        if self.transform:
            value = self.transform(value)
        self.widget.value = self.name.format(value=value)

    def set(self, count: str | int | float):
        self._render(count)


class Counter(Label):
    """Convenience wrapper for a counter label that can be incremented or updated."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.count: int | float = 0

    def update(self, count: int | float):
        self.count += count
        self._render(self.count)

    def inc(self):
        self.update(1)

    def clear(self):
        self.count = 0
        self._render(self.count)


def progress_widgets(epochs: int):
    """Get three common widgets for displaying training progress.

    Args:
        epochs (int): Number of epochs for the progress bar.

    Returns:
        ipywidget: `"progress"` for epoch progress (slider)
        ipywidget: `"train_loss"` for latest training loss (slider)
        ipywidget: `"val_loss"` for latest validation loss (slider)
    """
    progress_widget = ipywidgets.IntProgress(
        value=0,
        min=0,
        max=epochs,
        description=f"Epoch: 0/{epochs}",
    )
    training_loss_widget = ipywidgets.FloatProgress(
        value=1,
        min=0.0,
        max=1.0,
        description="Train: 1.0",
        bar_style="",
        style={"bar_color": "red"},
    )
    val_loss_widget = ipywidgets.FloatProgress(
        value=1,
        min=0.0,
        max=1.0,
        description="Val: 1.0",
        bar_style="",
        style={"bar_color": "red"},
    )

    return {
        "progress": progress_widget,
        "train_loss": training_loss_widget,
        "val_loss": val_loss_widget,
    }


def update_progress(epoch: int, epochs: int, widget: ipywidgets.IntProgress):
    """Update the epoch progress bar.

    Args:
        epoch (int): Current epoch (0-indexed).
        epochs (int): Total number of epochs.
        widget (ipywidgets.IntProgress): The progress bar widget to update.
    """
    widget.value = epoch + 1
    widget.description = f"Epoch: {epoch + 1}/{epochs}"


def update_loss(loss: float, widget: ipywidgets.FloatProgress, name: str = "Loss"):
    """Update a loss progress bar with color coding.

    Args:
        loss (float): Current loss value.
        widget (ipywidgets.FloatProgress): The loss bar widget to update.
        name (str, optional): Label for the bar. Defaults to "Loss".
    """
    widget.value = loss
    widget.description = f"{name}: {loss:.3f}"

    match loss:
        case loss if loss <= 0.1:
            widget.style = {"bar_color": "green"}
        case loss if loss < 0.3:
            widget.style = {"bar_color": "yellow"}
        case loss if loss < 0.5:
            widget.style = {"bar_color": "orange"}
        case _:
            widget.style = {"bar_color": "red"}
