import logging
import sys
from collections import defaultdict

import gensim

sys.path.append("..")
from utils.data_loader import Dataset  # noqa
from algorithms.base_recommender import BaseRecommender, RecommendResult  # noqa

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)


class Item2VecRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset) -> RecommendResult:
        # Parameters
        vector_size = 100
        n_epochs = 30
        window_size = 100
        use_skip_gram = 1
        use_hierarchial_softmax = 0
        words_min_count = 5

        train_data = dataset.train[dataset.train.rating >= 4]
        x_train = []
        for user_id, data in train_data.groupby("user_id"):
            x_train.append(
                data.sort_values("timestamp", ascending=False)["movie_id"].tolist()
            )

        model = gensim.models.Word2Vec(
            x_train,
            vector_size=vector_size,
            window=window_size,
            sg=use_skip_gram,
            hs=use_hierarchial_softmax,
            epochs=n_epochs,
            min_count=words_min_count,
        )

        pred_items = defaultdict(list)
        for user_id, data in train_data.groupby("user_id"):
            input_data = []
            for movie_id in data.sort_values("timestamp", ascending=False)[
                "movie_id"
            ].tolist():
                if movie_id in model.wv.key_to_index:
                    input_data.append(movie_id)

            if len(input_data) == 0:
                continue

            # TODO: Excluding user evalauted movies may be needed?
            pred_items[user_id] = [
                m[0] for m in model.wv.most_similar(input_data, topn=10)
            ]
        return RecommendResult(dataset.test.rating, pred_items)


if __name__ == "__main__":
    Item2VecRecommender().run_sample()
