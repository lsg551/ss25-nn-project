"""Metrics for IR ranking evaluation.

Metrics
=======

- MRR (mean reciprocal rank) - how high the first relevant result is ranked
- NDCG@k (normalized discounted cumulative gain) - evaluates ranking quality
  considering position and relevance scores; especially useful when there are
  multiple relevant results with graded relevance
- Recall@k – measures whether the correct answer is in the top-k results
- MAP (mean average precision) - useful when multiple relevant documents exist
"""


from collections.abc import Iterable, Sequence


def _to_bool_list(seq: Iterable) -> list[bool]:
    """Convert an iterable of truthy/falsy values into a list of bools.

    Accepts numbers (0/1), booleans, or numpy/torch scalar-likes. The order of
    items is preserved and expected to reflect the ranked order (best first).
    """
    return [bool(x) for x in seq]


def MRR(relevance_lists: Sequence[Iterable]) -> float:
    """Mean Reciprocal Rank (MRR).

    The MRR is a ranking metric that measures how high the first relevant
    result is ranked. It is the average of the reciprocal ranks of the first
    relevant item for each query.

    Expects, for each query, a ranked sequence of relevance indicators (best
    item first). Each inner sequence should contain truthy values (e.g., 1/0
    or True/False), where "truth" means the item is relevant.

    Behavior:
    - For each query, finds the rank (1-indexed) of the first relevant item;
        contributes 1/rank, or 0.0 if no relevant item is present.
    - Returns the mean over queries. For an empty input, returns 0.0.

    Example:
    >>> MRR([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    ... 0.5  # (1/2 + 1/3 + 0) / 3

    Args:
        relevance_lists (Sequence[Iterable]): A list (or other sequence) of
            ranked relevance indicator lists per query.
            Example: `[[1, 0, 0], [0, 0, 1], [0, 0, 0]]`

    Returns:
        float: The mean reciprocal rank (MRR) score.
    """

    if not relevance_lists:
        return 0.0

    total = 0.0
    for rels in relevance_lists:
        rel_bools = _to_bool_list(rels)
        rr = 0.0
        for idx, is_rel in enumerate(rel_bools, start=1):
            if is_rel:
                rr = 1.0 / idx
                break
        total += rr

    return total / len(relevance_lists)

    
def recall_at(relevance_lists: Sequence[Iterable], *, k: int) -> float:
    """Recall@k (aka Hit@k / Success@k).

    Measures whether at least one relevant item appears in the top-k results
    for each query and returns the mean over queries. For an empty input,
    returns 0.0.

    Example:
    >>> recall_at([[0, 1, 0], [0, 0, 1], [0, 0, 0]], k=10)
    ... 2/3

    Args:
        relevance_lists (Sequence[Iterable]): Ranked relevance indicator lists
            per query (best item first). Each inner sequence should contain truthy
            values (e.g., 1/0 or True/False), where truthy means the item is relevant.
        k (int): Cutoff rank. Must be >= 1.

    Returns:
        float: The recall@k score.
    """

    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer (>= 1)")

    if not relevance_lists:
        return 0.0

    hits = 0
    for rels in relevance_lists:
        rel_bools = _to_bool_list(rels)
        topk = rel_bools[:k]
        hits += 1 if any(topk) else 0

    return hits / len(relevance_lists)
