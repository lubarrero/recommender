import numpy as np
import pandas as pd
from typing import List
from scipy.sparse.linalg import svds

class Mi_SVD():

    def __init__(self):
        self._u = None
        self._s = None
        self._v = None
        self._mask = None
        self._u_index = None
        self._v_index = None


    def fit(self, data: pd.DataFrame, k: int) -> None:
        self._mask = data.notna().astype(int).values
        u, s, v = svds(data.fillna(0).values, k=k)
        self._u = u
        self._s = s
        self._v = v
        self._pred = np.dot(np.dot(u, np.diag(s)), v)
        self._u_index = data.index.tolist()
        self._v_index = data.columns.tolist()


    def predict(self, user: int, item: int):
        # obtengo el indice del id de user y de item
        u = self._u_index.index(user)
        i = self._v_index.index(item)
        return self._pred[u, i]
    

    def test(self, testset: pd.DataFrame, ind='userID', col='itemID'):
        predictions = [self.predict(ind, col) for (ind, col) in zip(testset[ind], testset[col])]
        return predictions
