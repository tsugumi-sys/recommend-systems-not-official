import sys
from collections import defaultdict

import pandas as pd
from surprise import SVD, Reader
from surprise import Dataset as SurpriseDataset

sys.path.append("..")
from utils.data_loader import Dataset  # noqa
from algorithms.base_recommender import BaseRecommender, RecommendResult  # noqa


class MFRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset) -> RecommendResult:
        # Parameters
        n_factors = 5
        min_rating_count = 100
        use_bias = True
        lr_all = 0.005
        n_epochs = 50

        train_data = dataset.train.groupby("movie_id").filter(
            lambda x: len(x["movie_id"]) >= min_rating_count
        )
        reader = Reader(rating_scale=(0.5, 5))
        train_data = SurpriseDataset.load_from_df(
            train_data[["user_id", "movie_id", "rating"]], reader
        ).build_full_trainset()
        mf = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, biased=use_bias)
        mf.fit(train_data)

        test_data = train_data.build_anti_testset(None)
        predictions = mf.test(test_data)
        pred_items = self.get_top_n(predictions)
        pred_ratings = pd.DataFrame.from_dict(
            [
                {"user_id": p.uid, "movie_id": p.iid, "pred_rating": p.est}
                for p in predictions
            ]
        )
        pred_ratings = dataset.test.merge(
            pred_ratings, on=["user_id", "movie_id"], how="left"
        )
        pred_ratings.fillna(dataset.train.rating.mean(), inplace=True)
        return RecommendResult(pred_ratings.pred_rating.to_numpy(), pred_items)

    def get_top_n(self, predictions, n=10):
        top_n = defaultdict(list)
        for user_id, movie_id, _, rating, _ in predictions:
            top_n[user_id].append((movie_id, rating))

        for user_id, ratings in top_n.items():
            ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[user_id] = [d[0] for d in ratings[:n]]
        return top_n


if __name__ == "__main__":
    MFRecommender().run_sample()
