import numpy as np
import pandas as pd
from typing import List


class BaselineUsers():

    def __init__(self):
        self._pred = None


    def fit(self, data: pd.DataFrame) -> None:
        pred = data.fillna(0)
        means = data.mean(axis = 1)
        for user in means.index:
            pred.loc[user] = means[user]
        self._pred = pred
        return 


    def predict(self, user: int, item: int):
        return self._pred.loc[user, item]
    

    def test(self, testset: pd.DataFrame, ind='userID', col='itemID'):
        predictions = [self.predict(ind, col) for (ind, col) in zip(testset[ind], testset[col])]
        return predictions


class BaselineItems():

    def __init__(self):
        self._pred = None


    def fit(self, data: pd.DataFrame) -> None:
        pred = data.fillna(0)
        medians = data.median()
        for item in medians.index:
            pred[item] = medians[item]
        self._pred = pred


    def predict(self, user: int, item: int):
        return self._pred.loc[user, item]
    

    def test(self, testset: pd.DataFrame, ind='userID', col='itemID'):
        predictions = [self.predict(ind, col) for (ind, col) in zip(testset[ind], testset[col])]
        return predictions
