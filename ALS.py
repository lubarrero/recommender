import numpy as np
import pandas as pd
from typing import List


class ALS():

    def __init__(self):
        self._u = None
        self._v = None
        self._mask = None
        self._u_index = None
        self._v_index = None


    def fit(self, data: pd.DataFrame, k: int, r_lambda: float, maxIter=100, stop_criteria=0.001) -> None:
        self._mask = data.notna().astype(int).values
        u, v = self._train_ALS(data.fillna(0).values, self._mask, k, r_lambda, maxIter, stop_criteria)
        self._u = u
        self._v = v
        self._pred = u.dot(v.T)
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


    def _get_error(self, R, mask, predicted):
        error = 0
        # Ignore nonzero elements
        error = np.square(R*mask - predicted*mask).sum()
        
        return(error/np.count_nonzero(mask))


    def _train_ALS(self, R, mask, k, r_lambda, maxIter, stop_criteria):
        # Initialize
        n, m = R.shape
        U = np.zeros((n,k))
        M = np.random.random((m,k))
        M[:,0] = [R[mask[:,i].nonzero(),i].mean() for i in range(m)]
        train_errors = []
        epsilons = []

        lambda_I = r_lambda * np.identity(k)
        
        for step in range(maxIter):

            for u in range(n):
                M_nonzero = M[mask[u]==True]
                a = M_nonzero.T.dot(M_nonzero) + lambda_I*sum(mask[u])
                v = M_nonzero.T.dot(R[u][mask[u]==True])
                U[u] = np.linalg.pinv(a).dot(v)

            for i in range(m):
                U_nonzero = U[mask[:,i]==True]
                a = U_nonzero.T.dot(U_nonzero) + lambda_I*sum(mask[:,i])
                v = U_nonzero.T.dot(R[:,i][mask[:,i]==True])
                M[i] = np.linalg.pinv(a).dot(v)
            
            predicted = U.dot(M.T)
            error = self._get_error(R, mask, predicted)
            
            # Convergence
            train_errors.append(error)
            epsilons.append(np.abs(train_errors[step-1]-error))
            if step > 2 and np.all(epsilon < stop_criteria for epsilon in epsilons[-3:]): # last 3 
                return(U, M)

        return(U, M)