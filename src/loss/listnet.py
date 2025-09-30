"""ListNet (top-1) loss for PyTorch.

Reference: Cao et al., "Learning to Rank: From Pairwise Approach to Listwise Approach" (2007):
https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf

This implementation uses the common top-1 approximation of ListNet:
- Convert model scores to a probability distribution with softmax over candidates
  (per query).
- Convert ground-truth labels to a target probability distribution (softmax over
  label values, optionally transformed) over the same candidate set.
- Minimize cross-entropy between these two distributions.
"""

from typing import Literal, Optional

import torch
import torch.nn.functional as F


def _transform_labels(
    labels: torch.Tensor, *, transform: Literal["identity", "exp2m1"]
) -> torch.Tensor:
    """Apply a transform to relevance labels before softmax normalization.

    Common choices:
    - identity: use labels as-is (e.g., 0/1 or graded relevance 0..4)
    - exp2m1: use 2^y - 1 to accentuate higher relevance grades
    """
    if transform == "identity":
        return labels
    
    if transform == "exp2m1":
        return torch.pow(2.0, labels) - 1.0



# NOTE: adapted from https://github.com/allegro/allRank/blob/master/allrank/models/losses/listNet.py
# but not yet tested
def listnet(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    ignore_index: Optional[float | int] = -1,
    label_transform: Literal["identity", "exp2m1"] = "identity",
    temperature: float = 1.0,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    """Compute the ListNet (top-1) loss for a batch of queries.

    Edge cases:
    - If a batch row has no valid candidates (all masked/ignored), its loss is
        0 and excluded from ``mean`` reduction (i.e., averaged over valid rows).

    Args:
        scores (torch.Tensor): Predicted scores (raw logits, one per candidate)
            with shape `(B, Cmax)`
        labels (torch.Tensor): Relevance labels with shape `(B, Cmax)`. This is
            the ground-truth signal to learn from. Unjudged positions can be
            marked with `ignore_index` and/or masked via `mask`.
        mask (torch.Tensor, optional): boolean mask of valid candidate positions
            with shape `(B, Cmax)`. Positions with `False` are ignored in both
            predicted and target distributions. If `None`, all positions are
            considered valid. This is useful when candidates are padded to a
            common length `Cmax`.
        ignore_index (float or int, optional): label value that indicates an
            unjudged/invalid position. These positions are ignored in both
            predicted and target distributions. If `None`, no label values are
            ignored. Default: -1. NOTE: `-1` is commonly used by annotators to
            indicate unjudged positions.
        label_transform (str): transformation to apply to labels before
            softmax normalization. One of:
            - "identity": use labels as-is (e.g., 0/1 or graded relevance)
            - "exp2m1": use 2^y - 1 to accentuate higher relevance grades
            Default: "identity"
        temperature (float): temperature for predicted distribution. Higher
            values produce a softer distribution. Default: 1.0
        reduction (str): reduction method for the final loss. One of:
            - "mean": average over valid batch rows (default)
            - "sum": sum over valid batch rows
            - "none": no reduction, return per-row losses with shape `(B,)`
    
    Raises:
        ValueError: if input shapes are incorrect or reduction is invalid

    Returns:
        torch.Tensor: The computed scalar loss. If reduction is `none`, a list
            (shape `(B,)`) of per-sample losses is returned. Otherwise, a single
            scalar is returned (mean or sum, according to `reduction`).
    """

    if scores.dim() != 2:
        raise ValueError(f"scores must be 2D (B, Cmax), got shape {tuple(scores.shape)}")

    if labels.shape != scores.shape:
        raise ValueError(
            f"labels must have same shape as scores, got {tuple(labels.shape)} vs {tuple(scores.shape)}"
        )

    device = scores.device
    B, Cmax = scores.shape # batch size, Cmax = max candidates per query

    # apply mask => valid positions
    if mask is None:
        eff_mask = torch.ones_like(scores, dtype=torch.bool, device=device)
    else:
        if mask.shape != scores.shape:
            raise ValueError(
                f"mask must have same shape as scores, got {tuple(mask.shape)} vs {tuple(scores.shape)}"
            )
        eff_mask = mask.to(torch.bool)

    if ignore_index is not None:
        eff_mask = eff_mask & (labels != ignore_index)

    # identify rows with at least one valid candidate
    valid_row = eff_mask.any(dim=1)  # (B,)
    num_valid_rows = int(valid_row.sum().item())

    # if no valid rows, return zero consistent with reduction
    if num_valid_rows == 0:
        if reduction == "none":
            return torch.zeros((B,), dtype=scores.dtype, device=device)
        return torch.tensor(0.0, dtype=scores.dtype, device=device)

    # Prepare logits for predicted distribution: mask invalid with -inf
    neg_inf = torch.finfo(scores.dtype).min if scores.is_floating_point() else -1e9
    pred_logits = scores / float(temperature)
    pred_logits = torch.where(eff_mask, pred_logits, torch.as_tensor(neg_inf, device=device, dtype=pred_logits.dtype))
    p_pred = F.softmax(pred_logits, dim=1)

    # Prepare target distribution from labels
    labels_f = labels.to(dtype=torch.get_default_dtype())
    t_logits = _transform_labels(labels_f, transform=label_transform)
    t_logits = torch.where(eff_mask, t_logits, torch.as_tensor(neg_inf, device=device, dtype=t_logits.dtype))
    p_tgt = F.softmax(t_logits, dim=1)

    # Cross-entropy: - sum_i p_tgt(i) * log p_pred(i), on valid positions only
    # Add tiny epsilon for numerical stability in log
    eps = 1e-12
    log_p_pred = torch.log(p_pred + eps)
    # Zero-out contributions from invalid positions
    per_pos = torch.where(eff_mask, p_tgt * log_p_pred, torch.zeros_like(log_p_pred))
    per_row_loss = -per_pos.sum(dim=1)  # (B,)

    if reduction == "none":
        # Zero-out rows that had no valid positions
        return torch.where(valid_row, per_row_loss, torch.zeros_like(per_row_loss))

    if reduction == "sum":
        return per_row_loss[valid_row].sum()

    # mean over valid rows only
    return per_row_loss[valid_row].mean()