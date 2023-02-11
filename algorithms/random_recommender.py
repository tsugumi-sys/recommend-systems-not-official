import sys
from collections import defaultdict

import numpy as np

from base_recommender import BaseRecommender, RecommendResult  # noqa

sys.path.append("..")
from utils.data_loader import Dataset, get_movielens_data  # noqa

rng = np.random.default_rng(123)


class RamdomRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        # Prediction Rating
        pred_ratings = rng.uniform(0.5, 5.0, len(dataset.test))
        # Recommend movies
        user_evaluated_movies = (
            dataset.train.groupby("user_id")
            .agg({"movie_id": list})["movie_id"]
            .to_dict()
        )
        unique_movie_ids = dataset.train.movie_id.unique()
        pred_items = defaultdict(list)
        for user_id in dataset.test_items.keys():
            pred_items[user_id] = rng.choice(
                unique_movie_ids[
                    [
                        movie_id in set(user_evaluated_movies[user_id])
                        for movie_id in unique_movie_ids
                    ]
                ],
                size=10,
            ).tolist()
        return RecommendResult(pred_ratings, pred_items)


if __name__ == "__main__":
    RamdomRecommender().run_sample()
