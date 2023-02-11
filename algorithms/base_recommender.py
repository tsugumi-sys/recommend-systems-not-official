import dataclasses
import sys
from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np

sys.path.append("..")
from utils.data_loader import DataLoader, Dataset  # noqa
from utils.metrics import MetricsCalculator  # noqa


@dataclasses.dataclass(frozen=True)
class RecommendResult:
    rating: np.ndarray
    items: Dict[int, List[int]]  # {user_id: [movie_ids]}


class BaseRecommender(ABC):
    @abstractmethod
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        pass

    def run_sample(self) -> None:
        movielens_dataset = DataLoader().load()
        # Calculate recommends and its metrics
        recommend_res = self.recommend(movielens_dataset)
        print(
            MetricsCalculator().metrics(
                pred_ratings=recommend_res.rating.tolist(),
                label_ratings=movielens_dataset.test.rating.tolist(),
                pred_items=recommend_res.items,
                label_items=movielens_dataset.test_items,
                k=10,
            )
        )
