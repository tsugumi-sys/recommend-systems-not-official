from collections import defaultdict, Counter
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import sys

sys.path.append("..")
from utils.data_loader import Dataset  # noqa
from algorithms.base_recommender import BaseRecommender, RecommendResult  # noqa


class AssociationRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset) -> RecommendResult:
        # Prepare user_id, movie_id, ratings matrix
        user_movie_ratings = dataset.train.pivot(
            index="user_id", columns="movie_id", values="rating"
        )
        user_movie_ratings[user_movie_ratings.isnull()] = 0
        user_movie_ratings[user_movie_ratings < 4] = 0
        user_movie_ratings[user_movie_ratings >= 4] = 1

        # Calculate lift
        movies_support = apriori(user_movie_ratings, min_support=0.1, use_colnames=True)
        lifts = association_rules(movies_support, metric="lift", min_threshold=1)
        lifts = lifts.sort_values(by="lift", ascending=False)
        antecedents, consequents = lifts["antecedents"].to_numpy(), lifts["consequents"]

        # Calculate recommends
        user_evaluated_movies = (
            dataset.train.groupby("user_id")
            .agg({"movie_id": list})["movie_id"]
            .to_dict()
        )
        recently_evaluated_movies = (
            dataset.train[dataset.train.rating > 4]
            .sort_values("timestamp", ascending=False)
            .groupby("user_id")
            .agg({"movie_id": list})["movie_id"]
            .to_dict()
        )
        pred_items = defaultdict(list)
        for user_id in dataset.test_items.keys():
            associated_movies = []
            for ant, conseq in zip(antecedents, consequents):
                if user_id not in recently_evaluated_movies:
                    break
                ant = [
                    movie_id
                    for movie_id in ant
                    if movie_id in set(recently_evaluated_movies[user_id][:5])
                ]
                if len(ant) > 0:
                    associated_movies += [
                        movie_id
                        for movie_id in conseq
                        if movie_id not in set(user_evaluated_movies[user_id])
                    ]
            recommends = []
            for movie_id, _ in Counter(associated_movies).most_common():
                recommends.append(movie_id)
                if len(recommends) == 10:
                    break
            print(user_id, recommends)
            pred_items[user_id] = recommends
        return RecommendResult(dataset.test.rating, pred_items)


if __name__ == "__main__":
    AssociationRecommender().run_sample()
