import sys
from collections import defaultdict

import implicit
import numpy as np
from scipy.sparse import lil_matrix

sys.path.append("..")
from utils.data_loader import Dataset  # noqa
from algorithms.base_recommender import BaseRecommender, RecommendResult  # noqa


class BPRRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset) -> RecommendResult:
        # Parameters
        n_factors = 5
        min_rating_count = 100
        n_epochs = 50
        rating_threshold = 4

        train_data = dataset.train.groupby("movie_id").filter(
            lambda x: len(x["movie_id"]) >= min_rating_count
        )
        train_data = train_data[dataset.train.rating >= rating_threshold]
        train_user_ids, train_movie_ids = (
            sorted(train_data.user_id.unique()),
            sorted(train_data.movie_id.unique()),
        )
        train_user_ids, train_movie_ids = np.array(train_user_ids), np.array(
            train_movie_ids
        )
        matrix = lil_matrix((len(train_user_ids), len(train_movie_ids)))
        for user_id, movie_id in zip(
            train_data.user_id.to_numpy(), train_data.movie_id.to_numpy()
        ):
            matrix[
                np.where(train_user_ids == user_id)[0][0],
                np.where(train_movie_ids == movie_id)[0][0],
            ] = 1.0

        model = implicit.bpr.BayesianPersonalizedRanking(
            factors=n_factors,
            iterations=n_epochs,
        )
        model.fit(matrix)

        recommends = model.recommend_all(matrix.tocsr())
        pred_items = defaultdict(list)
        for user_id in train_user_ids:
            pred_items[user_id] = train_movie_ids[
                recommends[np.where(train_user_ids == user_id)[0][0], :]
            ]
        # NOTE: Skip RMSE evaluation
        return RecommendResult(dataset.test.rating, pred_items)


if __name__ == "__main__":
    BPRRecommender().run_sample()
