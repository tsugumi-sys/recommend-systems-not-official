import dataclasses
import os
from typing import Dict, List

import numpy as np
import pandas as pd


###
# Configuration
###
class CFG:
    data_dir = "../data/ml-10M100K"
    movies_data_path = os.path.join(data_dir, "movies.dat")
    tags_data_path = os.path.join(data_dir, "tags.dat")
    ratings_data_path = os.path.join(data_dir, "ratings.dat")


###
# DataLoader
###
@dataclasses.dataclass(frozen=True)
class Dataset:
    train: pd.DataFrame
    test: pd.DataFrame
    test_items: Dict[int, List[int]]
    contents: pd.DataFrame


class DataLoader:
    def __init__(self, sampling: bool = True, num_test_items: int = 5):
        self.sampling = sampling
        self.num_test_items = num_test_items

    def load(self) -> Dataset:
        movielens_df = get_movielens_data(self.sampling)
        train_df, test_df = self._split_data(movielens_df)
        return Dataset(
            train=train_df,
            test=test_df,
            test_items=(
                test_df[test_df.rating >= 4]
                .groupby("user_id")
                .agg({"movie_id": list})["movie_id"]
                .to_dict()
            ),
            contents=(
                movielens_df.copy()
                .drop(columns=["Unnamed: 0", "user_id", "timestamp", "rating"])
                .drop_duplicates()
            ),
        )

    def _split_data(self, movielens_df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        movielens_df["rating_order"] = movielens_df.groupby("user_id")[
            "timestamp"
        ].rank(ascending=False, method="first")
        return (
            movielens_df[movielens_df["rating_order"] > self.num_test_items],
            movielens_df[movielens_df["rating_order"] <= self.num_test_items],
        )  # train, test


def get_movies_data(stdout_info: bool = True) -> pd.DataFrame:
    df = pd.read_csv(
        CFG.movies_data_path,
        names=["movie_id", "title", "genre"],
        sep="::",
        encoding="latin-1",
        engine="python",
    )
    df["genre"] = df.genre.apply(lambda x: x.split("|"))
    if stdout_info:
        print(df.head())
    return df


def get_tags_data(stdout_info: bool = True) -> pd.DataFrame:
    df = pd.read_csv(
        CFG.tags_data_path,
        names=["user_id", "movie_id", "tag", "timestamp"],
        sep="::",
        engine="python",
    )
    df["tag"] = df.tag.str.lower()
    if stdout_info:
        print(df.head())
        print("=" * 100)
        print("The number of unique tags:", len(df.tag.unique()))
        print("The number of tags records:", len(df))
        print("The number of movies with tags:", len(df.movie_id.unique()))
    return df


def get_ratings_data(sampling: bool = True, stdout_info: bool = True) -> pd.DataFrame:
    df = pd.read_csv(
        CFG.ratings_data_path,
        names=["user_id", "movie_id", "rating", "timestamp"],
        sep="::",
        engine="python",
    )
    if sampling:
        # Decreasing dataset size is effective for the faster verification of
        # each recommend algorithms as a first step.
        df = df.loc[df.user_id.isin(sorted(df.user_id.unique())[:1000])]
    if stdout_info:
        print(df.head())
    return df


def get_movielens_data(sampling: bool = True) -> pd.DataFrame:
    data_file_path = os.path.join(CFG.data_dir, "sampled_movielens_data.csv")
    if os.path.exists(data_file_path):
        return pd.read_csv(data_file_path)
    movie_df = get_movies_data(stdout_info=False)
    tags_df = get_tags_data(stdout_info=False).drop(columns=["timestamp", "user_id"])
    ratings = get_ratings_data(sampling, stdout_info=False)
    movielens_df = ratings.merge(movie_df, on="movie_id")
    movielens_df = movielens_df.merge(
        tags_df.groupby("movie_id").agg({"tag": list}), on="movie_id"
    )
    movielens_df.to_csv(data_file_path)
    return movielens_df


def statistics_of_movielens(sampling: bool = True) -> None:
    df = get_movielens_data(sampling)
    print(
        df.groupby("user_id")
        .agg({"movie_id": len})
        .agg({"movie_id": [min, max, np.mean, len]})
    )
    print(
        df.groupby("movie_id")
        .agg({"user_id": len})
        .agg({"user_id": [min, max, np.mean, len]})
    )
    print(df.groupby("rating").agg({"movie_id": len}))
