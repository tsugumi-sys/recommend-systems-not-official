import logging
import sys
from collections import defaultdict

import gensim
import numpy as np

sys.path.append("..")
from utils.data_loader import Dataset  # noqa
from algorithms.base_recommender import BaseRecommender, RecommendResult  # noqa

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)


class Word2VecRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset) -> RecommendResult:
        # Parameters
        vector_size = 100
        n_epochs = 20
        window_size = 100
        use_skip_gram = 1
        use_hierarchial_softmax = 0
        words_min_count = 5

        movie_content = dataset.contents.copy()
        movie_content["tag_genre"] = movie_content["tag"].fillna("").apply(
            list
        ) + movie_content["genre"].apply(list)
        movie_content["tag_genre"] = movie_content["tag_genre"].apply(
            lambda x: set(map(str, x))
        )

        tag_genre_data = movie_content.tag_genre.tolist()
        model = gensim.models.word2vec.Word2Vec(
            tag_genre_data,
            vector_size=vector_size,
            window=window_size,
            sg=use_skip_gram,
            hs=use_hierarchial_softmax,
            epochs=n_epochs,
            min_count=words_min_count,
        )

        movie_vectors = []
        tag_genre_in_model = set(model.wv.key_to_index.keys())

        movie_titles = []
        movie_ids = []
        for i, tag_genre in enumerate(tag_genre_data):
            input_tag_genre = set(tag_genre) & tag_genre_in_model
            if len(input_tag_genre) == 0:
                vector = np.random.randn(model.vector_size)
            else:
                vector = model.wv[input_tag_genre].mean(axis=0)
            movie_titles.append(movie_content.iloc[i]["title"])
            movie_ids.append(movie_content.iloc[i]["movie_id"])
            movie_vectors.append(vector)

        movie_vectors = np.array(movie_vectors)
        sum_vec = np.sqrt(np.sum(movie_vectors**2, axis=1))
        movie_norm_vectors = movie_vectors / sum_vec.reshape((-1, 1))

        x_train = dataset.train[dataset.train.rating > -4]
        user_evaluated_movies = (
            dataset.train.groupby("user_id")
            .agg({"movie_id": list})["movie_id"]
            .to_dict()
        )
        pred_items = defaultdict(list)
        movie_id2idx = dict(zip(movie_ids, range(len(movie_ids))))
        for user_id, data in x_train.groupby("user_id"):
            recent_movies = data.sort_values("timestamp", ascending=False)[
                "movie_id"
            ].tolist()[:5]
            recent_movie_idxes = [movie_id2idx[id] for id in recent_movies]
            user_vector = movie_norm_vectors[recent_movie_idxes].mean(axis=0)
            pred_items[user_id] = self.find_similar_items(
                user_vector,
                movie_norm_vectors,
                movie_ids,
                user_evaluated_movies[user_id],
            )
        return RecommendResult(dataset.test.rating, pred_items)

    def find_similar_items(
        self, vec, movie_norm_vectors, movie_ids, evaluated_movie_ids, topn=10
    ):
        score_vec = np.dot(movie_norm_vectors, vec)
        similar_indexes = np.argsort(-score_vec)
        similar_items = []
        for similar_idx in similar_indexes:
            similar_movie_id = movie_ids[similar_idx]
            if similar_movie_id not in evaluated_movie_ids:
                similar_items.append(similar_movie_id)
            if len(similar_items) == topn:
                break
        return similar_items


if __name__ == "__main__":
    Word2VecRecommender().run_sample()
