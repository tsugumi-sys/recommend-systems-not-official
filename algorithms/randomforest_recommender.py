import sys
from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestRegressor

sys.path.append("..")
from utils.data_loader import Dataset  # noqa
from algorithms.base_recommender import BaseRecommender, RecommendResult  # noqa


class RamdomForestRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset) -> RecommendResult:
        (x_train, y_train), x_test = self.preprocess(dataset)
        not_feature_cols = ["user_id", "movie_id"]
        # Training model
        model = RandomForestRegressor(n_jobs=-1, random_state=0)
        model.fit(x_train.drop(columns=not_feature_cols), y_train)
        # Prediction
        pred_ratings = model.predict(x_test.drop(columns=not_feature_cols))
        # Select recommends
        user_evaluated_movies = (
            dataset.train.groupby("user_id")
            .agg({"movie_id": list})["movie_id"]
            .to_dict()
        )
        pred_items = defaultdict(list)
        movie_cols = [c for c in x_train.columns if "movie" in c and c != "movie_id"]
        user_cols = [c for c in x_train.columns if "user" in c and c != "user_id"]
        for user_id in dataset.test_items.keys():
            x = x_train.loc[~x_train.movie_id.isin(user_evaluated_movies[user_id])]
            x = x[movie_cols + not_feature_cols]
            user_data = x_train.loc[x_train.user_id == user_id].iloc[0]
            for col in user_cols:
                x[col] = user_data[col]
            # NOTE: The columns order should be the same as x_train
            preds = model.predict(x[x_train.columns].drop(columns=not_feature_cols))
            pred_items[user_id] = x.movie_id.to_numpy()[
                np.argpartition(preds, -10)[-10:]
            ]
            pred_items[user_id]
        return RecommendResult(pred_ratings, pred_items)

    def preprocess(self, dataset):
        not_feature_cols = ["user_id", "movie_id"]
        x_train, x_test = (
            dataset.train[not_feature_cols].copy(),
            dataset.test[not_feature_cols].copy(),
        )
        y_train = dataset.train.rating.copy()
        aggs = ["min", "max", "mean"]
        # Calcalate features
        # TODO: Add other features
        user_features = dataset.train.groupby("user_id").agg({"rating": aggs})
        user_features.columns = ["userid-" + "-".join(c) for c in user_features.columns]
        movie_features = dataset.train.groupby("movie_id").agg({"rating": aggs})
        movie_features.columns = [
            "movieid-" + "-".join(c) for c in movie_features.columns
        ]
        # Add columns
        x_train = x_train.merge(user_features, on="user_id", how="left")
        x_train = x_train.merge(movie_features, on="movie_id", how="left")
        x_test = x_test.merge(user_features, on="user_id", how="left")
        x_test = x_test.merge(movie_features, on="movie_id", how="left")
        # Clean data
        for df in (x_train, x_test):
            # Fill NaN
            df.fillna(dataset.train.rating.mean(), inplace=True)
        return (x_train, y_train), x_test


if __name__ == "__main__":
    RamdomForestRecommender().run_sample()
