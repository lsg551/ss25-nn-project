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


def MRR(): ...


def recall_at(*, k: int): ...
