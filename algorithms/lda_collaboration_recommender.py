import logging
import sys
from collections import defaultdict

import gensim
from gensim.corpora.dictionary import Dictionary as GensimDict

sys.path.append("..")
from utils.data_loader import Dataset  # noqa
from algorithms.base_recommender import BaseRecommender, RecommendResult  # noqa

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)


class LDACollaborationRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset) -> RecommendResult:
        # Parameters
        n_topics = 50
        n_epochs = 30

        lda_data = []
        train_data = dataset.train[dataset.train.rating >= 4]
        for user_id, data in train_data.groupby("user_id"):
            lda_data.append(data["movie_id"].apply(str).tolist())

        common_dic = GensimDict(lda_data)
        common_corpus = [common_dic.doc2bow(text) for text in lda_data]

        model = gensim.models.LdaModel(
            common_corpus,
            id2word=common_dic,
            num_topics=n_topics,
            passes=n_epochs,
        )
        topics = model[common_corpus]

        user_evaluated_movies = (
            dataset.train.groupby("user_id")
            .agg({"movie_id": list})["movie_id"]
            .to_dict()
        )

        pred_items = defaultdict(list)
        for i, (user_id, data) in enumerate(train_data.groupby("user_id")):
            user_topic = sorted(topics[i], key=lambda x: -x[0])[0][0]
            topic_movies = model.get_topic_terms(user_topic, topn=len(dataset.contents))

            evaluated_movies = user_evaluated_movies[user_id]
            for token_id, score in topic_movies:
                movie_id = int(common_dic.id2token[token_id])
                if movie_id not in evaluated_movies:
                    pred_items[user_id].append(movie_id)
                if len(pred_items[user_id]) == 10:
                    break
        return RecommendResult(dataset.test.rating, pred_items)


if __name__ == "__main__":
    LDACollaborationRecommender().run_sample()
