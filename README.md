# Python実装 of 「推薦システム実践入門」, OREILLY, 2022

### **NOTE:** This is not official

[Official implementation.](https://github.com/oreilly-japan/RecommenderSystems)

## Performance of Algorithms

ユーザー1000人分のデータを用いた場合の各推薦アルゴリズムのメトリクス。

Move to `algorithm/` directory and run each algorithms' file.

| Algorithm               | RMSE  | Precision@K | Recall@K | Source                        |
| ----------------------- | ----- | ----------- | -------- | ----------------------------- |
| RandomRecommender       | 1.88  | 0.0         | 0.0      | `random_recommender.py`       |
| PopurarityRecommender   | 1.06  | 0.0         | 0.0      | `popularity_recommender.py`   |
| AssociationRecommender  | NaN   | 0.014       | 0.043    | `association_recommender.py`  |
| UMCRecommender          | 0.952 | 0.002       | 0.005    | `umc_recommender.py`          |
| RandomForestRecommender | 0.996 | 0.0002      | 0.004    | `randomforest_recommender.py` |
| SVDRecommender          | 1.04  | 0.020       | 0.065    | `svd_recommender.py`          |
| NMFRecommender          | 1.048 | 0.019       | 0.060    | `nmf_recommender.py`          |
| MFRecommender           | 1.027 | 0.010       | 0.034    | `mf_recommender.py`           |
| IMFRecommender          | NaN   | 0.023       | 0.073    | `imf_recommender.py`          |
