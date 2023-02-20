import logging
import sys
from collections import Counter, defaultdict

import gensim
import numpy as np
from gensim.corpora.dictionary import Dictionary as GensimDict

sys.path.append("..")
from utils.data_loader import Dataset  # noqa
from algorithms.base_recommender import BaseRecommender, RecommendResult  # noqa

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)


class LDAContentRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset) -> RecommendResult:
        # Parameters
        num_topics = 50
        n_epochs = 30

        movie_content = dataset.contents.copy()
        movie_content["tag_genre"] = movie_content.tag.fillna("").apply(
            list
        ) + movie_content.genre.apply(list)
        movie_content["tag_genre"] = movie_content.tag_genre.apply(
            lambda x: list(map(str, x))
        )

        tag_genre_data = movie_content.tag_genre.tolist()
        common_dict = GensimDict(tag_genre_data)
        common_corpus = [common_dict.doc2bow(text) for text in tag_genre_data]
        model = gensim.models.LdaModel(
            common_corpus,
            id2word=common_dict,
            num_topics=num_topics,
            passes=n_epochs,
        )
        topics = model[common_corpus]

        movie_topics, topic_scores = [], []
        for movie_idx, topic in enumerate(topics):
            t, s = sorted(topics[movie_idx], key=lambda x: -x[1])[0]
            movie_topics.append(t)
            topic_scores.append(s)
        movie_content["topic"] = movie_topics
        movie_content["topic_score"] = topic_scores

        # 1. Get top 10 movies that rated recently and higher of each users.
        # 2. Select the most popular topic of the movies as recomennding topic.
        # 3. Select the movies that are the same topic and not evaluated.
        pred_items = defaultdict(list)
        movie_ids = movie_content.movie_id.to_numpy()
        user_evaluated_movies = (
            dataset.train.groupby("user_id")
            .agg({"movie_id": list})["movie_id"]
            .to_dict()
        )
        for user_id, data in dataset.train[dataset.train.rating >= 4].groupby(
            "user_id"
        ):
            recent_rated_movies = data.sort_values("timestamp", ascending=False)[
                "movie_id"
            ].tolist()[:10]
            popular_topic = Counter(
                [
                    movie_topics[np.where(movie_ids == movie_id)[0][0]]
                    for movie_id in recent_rated_movies
                ]
            ).most_common(1)[0][0]
            print(
                Counter(
                    [
                        movie_topics[np.where(movie_ids == movie_id)[0][0]]
                        for movie_id in recent_rated_movies
                    ]
                )
            )
            same_topic_movies = (
                movie_content[movie_content.topic == popular_topic]
                .sort_values("topic_score", ascending=False)
                .movie_id.tolist()
            )
            evaluated_movies = set(user_evaluated_movies[user_id])
            recommends = []
            for movie_id in same_topic_movies:
                if movie_id not in evaluated_movies:
                    recommends.append(movie_id)
                if len(recommends) == 10:
                    break
            pred_items[user_id] = recommends
        return RecommendResult(dataset.test.rating, pred_items)


if __name__ == "__main__":
    LDAContentRecommender().run_sample()
