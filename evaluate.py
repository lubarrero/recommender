import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from compare import comparar_distancia_todos, comparar_posiciones_distintas_todos, comparar_distancia_top_n_todos
from metrics import getAP, getNDCG, getPrecisionRecall


METRICS_AT_KS = [1, 3, 5, 10, 25]
RELEVANT_MIN = 0.5


def _getRMSE(true_r: List[float], estimated: List[float]) -> int:
    assert len(true_r) == len(estimated), "Deben tener el mismo tamaño"
    mse = np.mean([(np.square(t - est)) for (t, est) in zip(true_r, estimated)])
    rmse = np.sqrt(mse)
    return rmse
    

def compareRMSE(pred_all: pd.DataFrame, plot=False):
    models = pred_all.columns[3:].tolist()
    RMSEs = []
    for model in models:
        RMSEs.append(_getRMSE(pred_all['rating'], pred_all[model]))
    compare_RMSE = pd.DataFrame({'model': models, 'RMSE': RMSEs})
    
    if plot:
        plt.figure(figsize=(8, 5))
        sns.barplot(x='model', y='RMSE', data=compare_RMSE).set(title='Test RMSE')
        plt.grid(axis='y')

    return compare_RMSE


def compareRankings(rank_all: pd.DataFrame, plot=False):
    models = rank_all.columns[2:].tolist()

    # ranking top5, top10 y top50 para test
    rank_top5 = rank_all.ranking.apply(lambda rank: rank[:5])
    rank_top10 = rank_all.ranking.apply(lambda rank: rank[:10])
    rank_top50 = rank_all.ranking.apply(lambda rank: rank[:50])

    # calculo 'distancia_top5', 'distancia_top10', 'distancia_top50' para todos los modelos
    compare_ranks = []
    for model in models:
        compare_ranks.append(comparar_distancia_todos(rank_all[model].apply(lambda rank: rank[:5]), \
                                                        rank_top5))
        compare_ranks.append(comparar_distancia_todos(rank_all[model].apply(lambda rank: rank[:10]), \
                                                        rank_top10))
        compare_ranks.append(comparar_distancia_todos(rank_all[model].apply(lambda rank: rank[:50]), \
                                                        rank_top50))

    error_ranks = pd.DataFrame(compare_ranks)
    # añado nombre de los modelos
    error_ranks['modelo'] = np.repeat(models, 3)
    # corrijo nombre de las comparaciones
    error_ranks['comparacion'] = ['distancia_top5', 'distancia_top10', 'distancia_top50'] * len(models)


    # calculo 'posiciones', 'distancia_todos', 'distancia_top10_todos', 'distancia_top50_todos' para todos los modelos
    compare_ranks = []
    for model in models:
        compare_ranks.append(comparar_posiciones_distintas_todos(rank_all[model], rank_all.ranking))
        compare_ranks.append(comparar_distancia_todos(rank_all[model], rank_all.ranking))
        compare_ranks.append(comparar_distancia_top_n_todos(rank_all[model], rank_all.ranking, 10))
        compare_ranks.append(comparar_distancia_top_n_todos(rank_all[model], rank_all.ranking, 50))

    # append al df con todos los errores
    error_ranks = error_ranks.append(pd.DataFrame(compare_ranks)).reset_index(drop=True)
    # añado nombre de los modelos que faltan
    error_ranks.loc[3*len(models):,'modelo'] = np.repeat(models, 4)

    error_ranks.sort_values(by=['comparacion', 'error_medio_acumulado'])

    if plot:
        for comparison in error_ranks.comparacion.unique():
            fig, axes = plt.subplots(1, 2, figsize=(15,5))
            sns.barplot(x='modelo', y='error_medio_medio', ax=axes[0],
                        data=error_ranks[error_ranks.comparacion==comparison]).\
                        set(title='Media de error_medio para '+ comparison)
            axes[0].grid(axis='y')
            sns.barplot(x='modelo', y='error_total_medio', ax=axes[1],
                        data=error_ranks[error_ranks.comparacion==comparison]).\
                        set(title='Media de error_total para '+ comparison)
            axes[1].grid(axis='y')

    return error_ranks


def getMetrics(rank_all: pd.DataFrame, test_melted: pd.DataFrame, plot=False):

    relevants = test_melted[test_melted.rating > RELEVANT_MIN].groupby('userID').\
        agg(relevant_items = ('itemID', lambda x: list(x)))
    relevants['relevant_length'] = relevants.relevant_items.apply(lambda r: len(r))
    
    # nombres de columnas de precision y de recall
    p_names = ['p' + str(k) for k in METRICS_AT_KS]
    r_names = ['r' + str(k) for k in METRICS_AT_KS]
    ap_names = ['ap' + str(k) for k in METRICS_AT_KS]
    ndcg_score_names = ['ndcg_score' + str(k) for k in METRICS_AT_KS]

    models = rank_all.columns[2:].tolist()
    metrics_results = pd.DataFrame(columns=['modelo', 'k', 'precision', 'recall', 'MAP', 'NDCG'])

    for name in models:
        metrics = getPrecisionRecall(rank_all[[name, 'length']], relevants, METRICS_AT_KS)
        getAP(metrics, METRICS_AT_KS)
        getNDCG(metrics, METRICS_AT_KS)
        metrics_results = pd.concat([metrics_results, 
                                    pd.DataFrame({
                                        'modelo': np.repeat(name, len(METRICS_AT_KS)),
                                        'k': METRICS_AT_KS,
                                        'precision': metrics[p_names].mean().values,
                                        'recall': metrics[r_names].mean().values,
                                        'MAP': metrics[ap_names].mean().values,
                                        'NDCG': metrics[ndcg_score_names].mean().values})
                                    ], ignore_index=True)

    if plot:
        plt.figure(figsize=(10, 7))
        line = sns.lineplot(x='recall', y='precision', hue='modelo', data=metrics_results, 
                            marker="o", palette="tab10").set_title('Curva Precision-Recall')
        for i, k in enumerate(metrics_results.k):
            plt.annotate(k, (metrics_results.recall[i]+0.003, metrics_results.precision[i]+0.003))
        plt.locator_params(axis='x', nbins=10)
        plt.legend(bbox_to_anchor=(1, 1), title='modelo')
        plt.grid()
        plt.show()

        plt.figure(figsize=(10, 7))
        sns.lineplot(x='k', y='MAP', hue='modelo', 
                    data=metrics_results, marker="o", palette="tab10").set_title('MAP')
        plt.legend(bbox_to_anchor=(1, 1), title='modelo')
        plt.locator_params(axis='x', nbins=20)
        plt.grid()
        plot.show()

        plt.figure(figsize=(10, 7))
        sns.lineplot(x='k', y='NDCG', hue='modelo', 
                    data=metrics_results, marker="o", palette="tab10").set_title('NDCG')
        plt.legend(bbox_to_anchor=(1, 1), title='modelo')
        plt.locator_params(axis='x', nbins=20)
        plt.grid()
        plot.show()

    return metrics_results
        