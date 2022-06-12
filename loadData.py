import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist,squareform

def loadMovies():
    movies = pd.read_csv("Data/movies.csv", usecols=['movieId', 'title'])
    movies = movies.set_index('movieId')
    return movies


def loadRatings():
    ratings = pd.read_csv("Data/ratings.csv", usecols=['userId', 'movieId', 'rating'])
    ratings.columns = ["userID", "itemID", "rating"]
    return ratings

def getRatingMatrix(ratings, ind='userID', col='itemID', val='rating'):
    mtx = ratings.pivot(index=ind, columns=col, values=val)
    return mtx
    

def getFilteredRatingMatrix(ratings, min_items, min_users, ind='userID', col='itemID', val='rating'):
    # obtengo items con count mayor al min items
    items_count = ratings[col].value_counts().sort_index()
    filtered_items = items_count[items_count > min_items].index
    # filtro del ratings los items y sobre eso obtengo los users con count mayor al min users
    users_count = ratings[ratings[col].isin(filtered_items)][ind].value_counts().sort_index()
    filtered_users = users_count[users_count > min_users].index
    # filtro en ratings los items y usuarios
    ratings = ratings[(ratings[col].isin(filtered_items)) & (ratings[ind].isin(filtered_users))]
    # creo matriz y la devuelvo    
    return getRatingMatrix(ratings, ind, col, val)


def splitRatings(rating_mtx, test_size):
    # inicializo train con los ratings y test con nan
    train_ratings = rating_mtx.values.copy() 
    test_ratings = np.nan * np.ones(rating_mtx.shape)
    # seleccion random de los indices a utilizar en test
    i,j = np.nonzero(rating_mtx.notna().astype(int).values)
    selected = np.random.choice(len(i), int(np.floor(test_size*len(i))), replace=False)
    # para los indices seleccionados, se incluye en test y se oculta en train
    for s in selected:
        test_ratings[i[s],j[s]] = train_ratings[i[s],j[s]].copy()
        train_ratings[i[s],j[s]] = np.nan
    # test y train son disjuntos
    assert(np.all(np.isnan(train_ratings * test_ratings)))
    # devuelvo df en lugar de array
    return(pd.DataFrame(train_ratings, columns=rating_mtx.columns, index=rating_mtx.index),\
        pd.DataFrame(test_ratings, columns=rating_mtx.columns, index=rating_mtx.index))


def normalizeRatings(rating_mtx):
    # calculo para cada user la media, 
    means = rating_mtx.mean(axis=1)
    # al rating de cada usuario, le resto su media
    normalized = rating_mtx.sub(means, axis=0) 
    # sobre los nuevos ratings, calculo para cada usuario, el rating minimo, el maximo y max-min
    norm_df = pd.DataFrame({'mean': means,
                            'min_r': normalized.min(axis=1), 
                            'max_r': normalized.max(axis=1)})
    norm_df['max-min'] = norm_df.apply(lambda row: row['max_r']-row['min_r'], axis=1)
    # normalizacion min max
    normalized = normalized.sub(norm_df['min_r'], axis=0).div(norm_df['max-min'], axis=0) 
    # devuelvo df normalizado y df con valores utilizados para la normalizacion
    return normalized, norm_df


def meltRatingMatrix(rating_mtx, ind='userID', col='itemID', val='rating'):
    melted = pd.melt(rating_mtx.reset_index(),
                     id_vars=[ind], value_vars=rating_mtx.columns.values,
                     value_name=val).dropna()
    return melted
