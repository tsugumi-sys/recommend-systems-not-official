import dataclasses
from typing import Dict, List

import numpy as np
from sklearn.metrics import mean_squared_error


###
# Metrics Class
###
@dataclasses.dataclass(frozen=True)
class Metrics:
    rmse: float
    precision_at_k: float
    recall_at_k: float


class MetricsCalculator:
    def metrics(
        self,
        pred_ratings: List[float],
        label_ratings: List[float],
        pred_items: Dict[int, List[int]],
        label_items: Dict[int, List[int]],
        k: int,
    ) -> Metrics:
        return Metrics(
            rmse=calc_rmse(pred_ratings, label_ratings),
            precision_at_k=calc_precition_at_k(pred_items, label_items, k),
            recall_at_k=calc_recall_at_k(pred_items, label_items, k),
        )


###
# RMSE
###
def calc_rmse(preds: List[float], labels: List[float]) -> float:
    return mean_squared_error(labels, preds, squared=False)


###
# Recall@K
###
def recall_at_k(pred_items: List[int], label_items: List[int], k: int) -> float:
    if len(label_items) == 0 or k == 0:
        return 0.0
    return (len(set(label_items) & set(pred_items[:k]))) / len(label_items)


def calc_recall_at_k(
    pred_items: Dict[int, List[int]], label_items: Dict[int, List[int]], k: int
) -> float:
    scores = []
    for user_id in pred_items.keys():
        scores.append(recall_at_k(pred_items[user_id], label_items[user_id], k))
    return np.mean(scores)


###
# Precition@k
###
def precition_at_k(pred_items: List[int], label_items: List[int], k: int) -> float:
    if k == 0:
        return 0.0
    return (len(set(label_items) & set(pred_items[:k]))) / k


def calc_precition_at_k(
    pred_items: Dict[int, List[int]], label_items: Dict[int, List[int]], k: int
) -> float:
    scores = []
    for user_id in pred_items.keys():
        scores.append(precition_at_k(pred_items[user_id], label_items[user_id], k))
    return np.mean(scores)
