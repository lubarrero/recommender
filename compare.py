import numpy as np
import pandas as pd
from typing import List


#----------- parte 1
def comparar_posiciones_distintas(rank1, rank2):
    # debemos asegurarnos de que ambos rankings tienen el mismo tamaño
    assert len(rank1) == len(rank2), "Los rankings deben tener el mismo tamaño"
    # compara con XOR bit a bit en el ranking, se suman los elementos distintos
    error_total = sum(np.array(rank1)^np.array(rank2)>0)
    metricas = {
        'error_total': error_total,
        'error_medio': error_total/len(rank1)
    }
    return metricas


def comparar_posiciones_distintas_todos(lista_rank1, lista_rank2):
    # debemos asegurarnos de que ambas listas de rankings tienen el mismo tamaño
    assert len(lista_rank1) == len(lista_rank2), "Las listas deben tener el mismo tamaño"

    resultados = []
    zipped = zip(lista_rank1, lista_rank2)
    for r1, r2 in zipped:
        resultados.append(comparar_posiciones_distintas(r1, r2))

    resultados = pd.DataFrame(resultados)
    metricas = {
        'comparacion': 'posicion',
        'error_total_acumulado': sum(resultados['error_total']),
        'error_total_medio': sum(resultados['error_total'])/len(lista_rank1),
        'error_medio_acumulado': sum(resultados['error_medio']),
        'error_medio_medio': sum(resultados['error_medio'])/len(lista_rank1)
    }
    return metricas#, resultados


# parte 2
# asumimos que puede pasar que un item este tan fuera de posicion en un ranking, que no existe en el otro ranking
# la penalización en ese caso sería el largo del ranking + 1, salvo que se asigne un valor manualmente
def comparar_distancia(rank1, rank2, penalty=None):
    # debemos asegurarnos de que ambos rankings tienen el mismo tamaño
    assert len(rank1) == len(rank2), "Los rankings deben tener el mismo tamaño"
    # valor por defecto de penalidad: el largo del ranking + 1
    if penalty is None: penalty = len(rank1)+1

    # el error se calcula sumando para cada elemento del ranking la diferencia entre los indices
    # si no se encuentra en rank2, se suma la penalidad
    error = 0
    for i in range(len(rank1)):
        error += abs(i-np.where(rank2==rank1[i])[0][0]) if rank1[i] in rank2 else penalty

    metricas = {
        'error_total': error,
        'error_medio': error/len(rank1)
    }
    return metricas


def comparar_distancia_todos(lista_rank1, lista_rank2, lista_penalties=None):
    # debemos asegurarnos de que las listas de rankings tienen el mismo tamaño
    assert len(lista_rank1) == len(lista_rank2), "Las listas deben tener el mismo tamaño"
    if lista_penalties is None: 
        lista_penalties = [None]*len(lista_rank1)
    else:
        assert len(lista_rank1) == len(lista_penalties), "Las listas deben tener el mismo tamaño"

    resultados = []
    zipped = zip(lista_rank1, lista_rank2, lista_penalties)
    for r1, r2, p in zipped:
        resultados.append(comparar_distancia(r1, r2, p))

    resultados = pd.DataFrame(resultados)
    metricas = {
        'comparacion': 'distancia',
        'error_total_acumulado': sum(resultados['error_total']),
        'error_total_medio': sum(resultados['error_total'])/len(lista_rank1),
        'error_medio_acumulado': sum(resultados['error_medio']),
        'error_medio_medio': sum(resultados['error_medio'])/len(lista_rank1)
    }
    return metricas#, resultados


# parte 3
# comparar distancia top n de dos rankings completos
def comparar_distancia_top_n(rank1, rank2, n):
    # debemos asegurarnos de que ambos rankings tienen el mismo tamaño
    assert len(rank1) == len(rank2), "Los rankings deben tener el mismo tamaño"

    # para los n primero elementos del rank1 se busca la distancia con el rank2
    error = 0
    for i in range(min(n, len(rank1))): # si n es mayor que los rankings, comparo el ranking
        error += abs(i-np.where(rank2==rank1[i])[0][0]) 

    metricas = {
        'error_total': error,
        'error_medio': error/len(rank1)
    }
    return metricas

def comparar_distancia_top_n_todos(lista_rank1, lista_rank2, n):
    # debemos asegurarnos de que las listas de rankings tienen el mismo tamaño
    assert len(lista_rank1) == len(lista_rank2), "Las listas deben tener el mismo tamaño"

    resultados = []
    zipped = zip(lista_rank1, lista_rank2)
    for r1, r2 in zipped:
        resultados.append(comparar_distancia_top_n(r1, r2, n))

    resultados = pd.DataFrame(resultados)
    metricas = {
        'comparacion': 'distancia_top'+ str(n) + '_todos',
        'error_total_acumulado': sum(resultados['error_total']),
        'error_total_medio': sum(resultados['error_total'])/len(lista_rank1),
        'error_medio_acumulado': sum(resultados['error_medio']),
        'error_medio_medio': sum(resultados['error_medio'])/len(lista_rank1)
    }
    return metricas#, resultados
