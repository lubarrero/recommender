import numpy as np
import pandas as pd
from typing import List
import math
from sklearn.metrics import ndcg_score


def getPrecisionRecall(rankings: pd.DataFrame, relevants: pd.DataFrame, ks: List[int]): 
    df = relevants.join(rankings)
    df.rename(columns={df.columns[-2]: 'ranking'}, inplace=True)
    for k in ks:
        df['valid_k'] = (df.length >= k)
        # obtengo top k del ranking
        df['rank_k'] = df.ranking.apply(lambda rank: rank[:k])
        # obtengo cantidad de elementos relevantes del top k
        df['ranked_relevant'] = df.apply(lambda row: len([i for i in row['rank_k'] if i in row['relevant_items']]), axis = 1)
        # calculo precision y recall
        df['p'+str(k)] = df['ranked_relevant']/k
        df['r'+str(k)] = df['ranked_relevant']/df['relevant_length']
        # si k es mayor al largo del ranking, le asigno nan
        df.loc[~df.valid_k,'p'+str(k)] = np.nan
        df.loc[~df.valid_k,'r'+str(k)] = np.nan
    # elimino columnas innecesarias
    df.drop(['valid_k', 'rank_k', 'ranked_relevant'], axis = 1, inplace=True)
    return df


def getAP(df, ks):
    # genero todas las precisiones de k = 1 hasta k = max(ks)
    df['precisions'] = [[]] * df.shape[0]
    for k in range(1, max(ks)+1):
        df['valid_k'] = (df['length'] >= k)
        # obtengo top k del ranking
        df.loc[df.valid_k,'rank_k'] = df.ranking.apply(lambda rank: rank[:k])
        # obtengo cantidad de elementos relevantes del top k
        df.loc[df.valid_k,'ranked_relevant'] = df.loc[df.valid_k,:].apply(lambda row: len([i for i in row['rank_k'] if i in row['relevant_items']]), axis = 1)
        # calculo si el elemento k del ranking es relevante
        df.loc[df.valid_k,'k_is_relevant'] = df.loc[df.valid_k,:].apply(lambda row: int(row['rank_k'][k-1] in row['relevant_items']), axis = 1)
        # p = 0 si el elemento k no es relevante | p = precision si k es relevante
        df.loc[df.valid_k,'p'] = df.loc[df.valid_k,:].apply(lambda row: (min(row['k_is_relevant'], row['ranked_relevant']/k)), axis=1)
        # concateno vector de precisiones
        df.loc[df.valid_k,'precisions'] = df.loc[df.valid_k,:].apply(lambda row: row['precisions'] + [row['p']], axis = 1)

    # calculo AP para todos los ks
    for k in ks:
        df['valid_k'] = (df['length'] >= k)
        # calculo AP como la suma de las k precisiones / min(k, cant elem relevantes)
        df['ap'+str(k)] = df.apply(lambda row: sum(row['precisions'][:k])/min(k, row['relevant_length']), axis=1) 
        # si k es mayor al largo del ranking, le asigno nan
        df.loc[~df.valid_k, 'ap'+str(k)]= np.nan
    # elimino columnas innecesarias
    df.drop(['valid_k', 'rank_k', 'ranked_relevant', 'k_is_relevant', 'p', 'precisions'], axis = 1, inplace=True)


def getNDCG(df, ks):
    max_k = max(ks)
    # calculo DCG
    df['DCG'] = [[]] * df.shape[0]
    df['IDCG'] = df['relevant_length'].apply(lambda i: [1/ math.log2(k+1) for k in range(1, min(max_k, i)+1)])
    for k in range(1, max_k+1):
        df['valid_k'] = (df['relevant_length'] >= k)
        # obtengo top k del ranking
        df.loc[df.valid_k,'rank_k'] = df.loc[df.valid_k,'ranking'].apply(lambda rank: rank[:k])
        # calculo si el elemento k del ranking es relevante
        df.loc[df.valid_k,'k_is_relevant'] = df.loc[df.valid_k,:].apply(lambda row: int(row['rank_k'][k-1] in row['relevant_items']), axis = 1)
        # calculo DCG para el elemento k
        df.loc[df.valid_k,'k_dcg'] = df.loc[df.valid_k,'k_is_relevant'] / math.log2(k+1)
        # concateno vector de DCG
        df.loc[df.valid_k,'DCG'] = df.loc[df.valid_k,:].apply(lambda row: row['DCG'] + [row['k_dcg']], axis = 1)

    # calculo ndcg_score
    for k in ks:
        df['valid_k'] = ((df['relevant_length'] >= k) & (df['relevant_length'] > 1))
        # por defecto se asigna nan
        df['ndcg_score'+str(k)] = np.nan
        # calculo ndcg_score entre vector los k elementos de DCG y IDCG
        df.loc[df.valid_k,'ndcg_score'+str(k)] = df.loc[df.valid_k,:].\
            apply(lambda row: ndcg_score(np.asarray([row['DCG']]),np.asarray([row['IDCG']]), k=k), axis=1) 
    # elimino columnas innecesarias
    df.drop(['valid_k', 'rank_k', 'k_is_relevant', 'k_dcg'], axis = 1, inplace=True)
