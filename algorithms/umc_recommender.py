import sys
from collections import defaultdict

import numpy as np
from surprise import Dataset as SurpriseDataset
from surprise import KNNWithMeans, Reader

sys.path.append("..")
from utils.data_loader import Dataset  # noqa
from algorithms.base_recommender import BaseRecommender, RecommendResult  # noqa


class UMCRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset) -> RecommendResult:
        reader = Reader(rating_scale=(0.5, 5))
        train_data = SurpriseDataset.load_from_df(
            dataset.train[["user_id", "movie_id", "rating"]], reader
        ).build_full_trainset()
        knn = KNNWithMeans(
            k=30, min_k=1, sim_options={"name": "pearson", "user_based": True}
        )
        knn.fit(train_data)
        predictions = knn.test(train_data.build_anti_testset(None))
        pred_items = self._get_top_n(predictions)
        rating_pred = []
        mean_rating = dataset.train.rating.mean()
        train_user_ids = set(dataset.train.user_id.unique().tolist())
        train_movie_ids = set(dataset.train.movie_id.unique().tolist())
        for user_id, movie_id in zip(dataset.test.user_id, dataset.test.movie_id):
            if user_id not in train_user_ids or movie_id not in train_movie_ids:
                rating_pred.append(mean_rating)
                continue
            rating_pred.append(knn.predict(uid=user_id, iid=movie_id).est)
        return RecommendResult(np.array(rating_pred), pred_items)

    def _get_top_n(self, predictions, n: int = 10):
        top_n = defaultdict(list)
        for user_id, movie_id, _, rating, _ in predictions:
            top_n[user_id].append((movie_id, rating))
        for user_id, ratings in top_n.items():
            ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[user_id] = [i[0] for i in ratings[:n]]
        return top_n


if __name__ == "__main__":
    UMCRecommender().run_sample()
