import pandas as pd
import numpy as np
from ALS import ALS
from SVD import SVD

    
def known(ratings: pd.DataFrame, n: int) -> pd.DataFrame:
    items = ratings.columns.name
    rank = []
    for u in ratings.index:
        rank.append(np.array(ratings.loc[u].dropna().reset_index().sort_values(by=[u, items],\
            ascending=[False, True])[items][:n]))
    rankings = pd.DataFrame({'ranking': rank}, index = ratings.index).rename_axis('userId')
    return rankings


def baseline(ratings: pd.DataFrame, n: int) -> pd.DataFrame:
    items = ratings.columns.name
    rank = np.array(ratings.sum().reset_index().sort_values(by=[0, items], ascending=[False,True])[items][:n])
    rankings = pd.DataFrame({'ranking': [rank for i in range(len(ratings))]}, index = ratings.index).rename_axis('userId')
    return rankings


def als(ratings: pd.DataFrame, k: int, r_lambda: float , n: int) -> pd.DataFrame:
    als = ALS()
    als.fit(ratings, k, r_lambda)
    rankings = als.predict_for_multiple(ratings.index.tolist(), n)
    return rankings


def svd(ratings: pd.DataFrame, k: int, n: int) -> pd.DataFrame:
    svd = SVD()
    svd.fit(ratings, k)
    rankings = svd.predict_for_multiple(ratings.index.tolist(), n)
    return rankings


