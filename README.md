# Python実装 of 「推薦システム実践入門」, OREILLY, 2022

### **NOTE:** This is not official

[Official implementation.](https://github.com/oreilly-japan/RecommenderSystems)

## Performance of Algorithms

ユーザー1000人分のデータを用いた場合の各推薦アルゴリズムのメトリクス参考値。

Move to `algorithm/` directory and run each algorithms' file.

| Algorithm                    | RMSE  | Precision@K | Recall@K | Source                             |
| ---------------------------- | ----- | ----------- | -------- | ---------------------------------- |
| RandomRecommender            | 1.88  | 0.0         | 0.0      | `random_recommender.py`            |
| PopurarityRecommender        | 1.06  | 0.0         | 0.0      | `popularity_recommender.py`        |
| AssociationRecommender       | NaN   | 0.014       | 0.043    | `association_recommender.py`       |
| UMCRecommender               | 0.952 | 0.002       | 0.005    | `umc_recommender.py`               |
| RandomForestRecommender      | 0.996 | 0.0002      | 0.004    | `randomforest_recommender.py`      |
| SVDRecommender               | 1.04  | 0.020       | 0.065    | `svd_recommender.py`               |
| NMFRecommender               | 1.048 | 0.019       | 0.060    | `nmf_recommender.py`               |
| MFRecommender                | 1.027 | 0.010       | 0.034    | `mf_recommender.py`                |
| IMFRecommender               | NaN   | 0.023       | 0.073    | `imf_recommender.py`               |
| BPRRecommender               | NaN   | 0.022       | 0.069    | `bpr_recommender.py`               |
| FMRecommender                | 1.055 | 0.013       | 0.041    | `fm_recommender.py`                |
| LDAContentRecommender        | NaN   | 0.0         | 0.0      | `lda_content_recommender.py`       |
| LDACollaboprationRecommender | NaN   | 0.018       | 0.057    | `lda_collaboration_recommender.py` |
| Word2VecRecommender          | NaN   | 0.001       | 0.003    | `word2vec_recommender.py`          |
| Item2VecRecommender          | NaN   | 0.027       | 0.085    | `item2vec_recommender.py`          |

## 未実装のアルゴリズム

- RNN (Session-based recommendations with RNN, Balaz Hidasi et al, 2015)
- item2vec (Neural item embedding for collaborative filtering, Oren Barkan and
  Noam Koenigsten, 2016 & Mihajlo E-commerce ub your inbox: Product
  recommendations at scale, Grbovic et al, 2015)
- BERT (BERT4Rec: Sequential recommendation with bidirectional encoder
  representations from transformer, Fei Sun et al, 2019)
- Nerural Collaborative Filtering (Xiangnan He, et al, 2017)
- Wide and Deep (Heng-Tze Cheng et al, 2016, Google)

### DeepLearning for recommendationsライブラリ

- Recommenders (microsoft)
- Spotlight (maciejkula)
- RecBole (recbole)

### 注目論文

- Are We Really Making Much Progress? A Worring Analysis of Recent Neural
  Recommendation Approaches (Maurizio Ferrari Dacrema et al, 2019)
- A Survey on Contextual Multi-armed Bandit (Arxiv)
