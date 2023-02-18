import sys
from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd
import xlearn as xl
from sklearn.preprocessing import OneHotEncoder

sys.path.append("..")
from utils.data_loader import Dataset  # noqa
from algorithms.base_recommender import BaseRecommender, RecommendResult  # noqa


class FMRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset) -> RecommendResult:
        # Parameters
        n_factors = 10
        min_rating_count = 200
        n_epochs = 50
        learning_rate = 0.01
        # use_side_information = True

        x_train, y_train, onehot_encoder = self._preprocess(dataset, min_rating_count)
        model = xl.FMModel(
            task="reg",
            metric="rmse",
            lr=learning_rate,
            opt="sgd",
            k=n_factors,
            epoch=n_epochs,
        )
        model.fit(x_train, y_train, is_lock_free=False)

        pred_matrix, unique_user_ids, unique_movie_ids = self._predict(
            model, onehot_encoder, dataset, min_rating_count
        )
        mean_rating = dataset.train.rating.mean()
        pred_ratings = []
        for user_id, movie_id in zip(
            dataset.test.user_id.to_numpy(), dataset.test.movie_id.to_numpy()
        ):
            try:
                pred_ratings.append(
                    pred_matrix[
                        np.where(unique_user_ids == user_id)[0][0],
                        np.where(unique_movie_ids == movie_id)[0][0],
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
        for user_id in unique_user_ids:
            user_pred_matrix = pred_matrix[
                np.where(unique_user_ids == user_id)[0][0], :
            ]
            user_not_evaluated_movies = [
                i not in set(user_evaluated_movies[user_id]) for i in unique_movie_ids
            ]
            user_pred_matrix = user_pred_matrix[user_not_evaluated_movies]
            target_movie_ids = unique_movie_ids[user_not_evaluated_movies]
            if len(target_movie_ids) < 10:
                pred_items[user_id] = target_movie_ids
            else:
                pred_items[user_id] = target_movie_ids[
                    np.argpartition(user_pred_matrix, -10)[-10:]
                ]
        return RecommendResult(np.array(pred_ratings), pred_items)

    def _predict(
        self,
        trained_model,
        onehot_encoder: OneHotEncoder,
        dataset: Dataset,
        min_rating_count: int,
    ):
        # Use only enough reviews dataset
        test_data = dataset.train.groupby("movie_id").filter(
            lambda x: len(x["movie_id"]) >= min_rating_count
        )
        unique_user_ids, unique_movie_ids = sorted(test_data.user_id.unique()), sorted(
            test_data.movie_id.unique()
        )
        user_average_rating = dataset.train.groupby("user_id").agg({"rating": "mean"})
        user_average_rating.reset_index(inplace=True)
        user_average_rating.rename(columns={"rating": "mean_rating"}, inplace=True)

        x_test = pd.DataFrame(
            product(unique_user_ids, unique_movie_ids), columns=["user_id", "movie_id"]
        )
        # Set User average ratings
        user_average_rating = (
            dataset.train[["user_id", "rating"]]
            .groupby("user_id")
            .agg({"rating": "mean"})
        )
        user_average_rating.reset_index(inplace=True)
        user_average_rating.rename(columns={"rating": "mean_rating"}, inplace=True)
        x_test = x_test.merge(user_average_rating, on="user_id", how="left")
        user_average_rating = x_test.mean_rating.to_numpy()
        x_test.drop(columns=["mean_rating"], inplace=True)

        x_test.user_id = x_test.user_id.astype(str)
        x_test.movie_id = x_test.movie_id.astype(str)
        movie_tags = dataset.train[["movie_id", "tag"]]
        movie_tags.drop_duplicates(inplace=True)
        movie_tags.movie_id = movie_tags.movie_id.astype(str)
        x_test = x_test.merge(movie_tags, on="movie_id", how="left")
        x_test = onehot_encoder.transform(x_test.to_numpy()).toarray()

        x_test = np.concatenate((x_test, user_average_rating.reshape(-1, 1)), axis=1)
        return (
            trained_model.predict(x_test).reshape(
                len(unique_user_ids), len(unique_movie_ids)
            ),
            np.array(unique_user_ids),
            np.array(unique_movie_ids),
        )

    def _preprocess(self, dataset: Dataset, min_rating_count: int):
        # Use only enough reviews dataset
        train_data = dataset.train.groupby("movie_id").filter(
            lambda x: len(x["movie_id"]) >= min_rating_count
        )
        # Convert int columns as string for one-hot encoding
        for col in ["user_id", "movie_id"]:
            train_data[col] = train_data[col].astype(str)
        # Calcularte user based average rating score
        user_average_rating = dataset.train.groupby("user_id").agg({"rating": "mean"})
        user_average_rating.reset_index(inplace=True)
        user_average_rating.rename(columns={"rating": "mean_rating"}, inplace=True)
        user_average_rating.user_id = user_average_rating.user_id.astype(str)
        train_data = train_data.merge(
            user_average_rating,
            on="user_id",
        )
        x_train = train_data[["user_id", "movie_id", "tag", "mean_rating"]]
        y_train = train_data.rating.to_numpy()
        # Apply one-hot encoding for user id, movie id and tag
        onehot_encoder = OneHotEncoder(handle_unknown="ignore")
        encoded_train_data = onehot_encoder.fit_transform(
            x_train.drop(columns="mean_rating").to_numpy()
        ).toarray()
        x_train = np.concatenate(
            (encoded_train_data, x_train.mean_rating.to_numpy().reshape(-1, 1)),
            axis=1,
        )
        return x_train, y_train, onehot_encoder


if __name__ == "__main__":
    FMRecommender().run_sample()
