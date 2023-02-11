import sys
from collections import defaultdict

import numpy as np

sys.path.append("..")
from utils.data_loader import Dataset  # noqa
from algorithms.base_recommender import BaseRecommender, RecommendResult  # noqa


class PopularityRecommender(BaseRecommender):
    def __init__(self, minimum_rating_counts: int = 100):
        self.minimum_rating_counts = minimum_rating_counts

    def recommend(self, dataset: Dataset) -> RecommendResult:
        rating_stats = dataset.train.groupby("movie_id").agg(
            {"rating": [np.size, np.mean]}
        )
        pred_ratings = dataset.test.merge(
            rating_stats, on="movie_id", how="left"
        ).fillna(0)[("rating", "mean")]
        user_evaluated_movies = (
            dataset.train.groupby("user_id")
            .agg({"movie_id": list})["movie_id"]
            .to_dict()
        )
        rating_stats = rating_stats.loc[
            rating_stats[("rating", "size")] > self.minimum_rating_counts
        ].sort_values(by=[("rating", "mean")])
        pred_items = defaultdict(list)
        for user_id in dataset.test_items.keys():
            pred_items[user_id] = rating_stats.index.to_numpy()[
                [
                    movie_id in set(user_evaluated_movies[user_id])
                    for movie_id in rating_stats.index
                ]
            ][:10]
        return RecommendResult(pred_ratings, pred_items)


if __name__ == "__main__":
    PopularityRecommender(minimum_rating_counts=100).run_sample()
