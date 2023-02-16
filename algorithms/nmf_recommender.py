import sys
from collections import defaultdict

import numpy as np
from sklearn.decomposition import NMF

sys.path.append("..")
from utils.data_loader import Dataset  # noqa
from algorithms.base_recommender import BaseRecommender, RecommendResult  # noqa


class NMFRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset) -> RecommendResult:
        user_movie_matrix = dataset.train.pivot(
            index="user_id", columns="movie_id", values="rating"
        )
        mean_rating = dataset.train.rating.mean()
        user_movie_matrix = user_movie_matrix.fillna(mean_rating)
        train_user_ids, train_movie_ids = (
            user_movie_matrix.index.to_numpy(),
            user_movie_matrix.columns.to_numpy(),
        )
        nmf = NMF(n_components=5)
        nmf.fit(user_movie_matrix.to_numpy())
        P, Q = nmf.fit_transform(user_movie_matrix.to_numpy()), nmf.components_

        pred_matrix = np.dot(P, Q)
        pred_ratings = []
        for user_id, movie_id in zip(
            dataset.test.user_id.to_numpy(), dataset.test.movie_id.to_numpy()
        ):
            try:
                pred_ratings.append(
                    pred_matrix[
                        np.where(train_user_ids == user_id)[0][0],
                        np.where(train_movie_ids == movie_id)[0][0],
                    ]
                )
            except IndexError:
                pred_ratings.append(mean_rating)
        user_evaluated_movies = (
            dataset.train.groupby("user_id")
            .agg({"movie_id": list})["movie_id"]
            .to_dict()
        )
        pred_items = defaultdict(list)
        for user_id in dataset.test_items.keys():
            try:
                user_pred_matrix = pred_matrix[
                    np.where(train_user_ids == user_id)[0][0], :
                ]
                user_not_evaluated_movies = [
                    i not in set(user_evaluated_movies[user_id])
                    for i in train_movie_ids
                ]
                user_pred_matrix = user_pred_matrix[user_not_evaluated_movies]
                pred_items[user_id] = train_movie_ids[user_not_evaluated_movies][
                    np.argpartition(user_pred_matrix, -10)[-10:]
                ]
            except IndexError:
                continue
        return RecommendResult(np.array(pred_ratings), pred_items)


if __name__ == "__main__":
    NMFRecommender().run_sample()
