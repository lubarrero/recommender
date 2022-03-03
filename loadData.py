from numpy import NaN
import pandas as pd
from scipy.spatial.distance import pdist,squareform

def loadMovies():
    movies = pd.read_csv("Data/movies.csv", usecols=['movieId', 'title'])
    movies = movies.set_index('movieId')
    #print(movies.head())
    return(movies)


def loadRatings():
    ratings = pd.read_csv("Data/ratings.csv", usecols=['userId', 'movieId', 'rating'])
    #print(ratings.head())
    return(ratings)


def getRatingMatrix(ratings):
    mtx = ratings.pivot(index='userId', columns='movieId', values='rating')
    means = mtx.mean(axis=1)
    mtx = mtx.sub(means, axis = 0) # normalizacion
    mtx = mtx.fillna(0)
    #print(mtx.head())
    return(mtx, means)


def getSimilarityMatrix(mtx):
    sim_mtx = pd.DataFrame(squareform(1 - pdist(mtx, 'cosine')), columns=mtx.index, index=mtx.index)
    print(sim_mtx.head())
    return(sim_mtx)


# set of k users similary to user_x that ranked item i
def getSimilarUsers(mtx, sim_mtx, k, user_x, i):
    ranked_i = mtx.loc[mtx[i].apply(lambda x: x!=0)].index.tolist()
    if len(ranked_i) > 0:
        top_k = sim_mtx.loc[ranked_i][user_x].sort_values(ascending=False).iloc[:k].index.tolist()
    else:
        top_k = []
    return(top_k)


def predictItemRating(mtx, sim_mtx, k, user_x, i):
    neighborhood = getSimilarUsers(mtx, sim_mtx, k, user_x, i)
    sum_similarity = sim_mtx.loc[neighborhood][user_x].sum()
    if len(neighborhood) > 0 and sum_similarity != 0:
        rating = (sim_mtx.loc[neighborhood][user_x]*mtx.loc[neighborhood][i]).values.sum()
        rating = rating/sum_similarity
    else:
        rating = mtx.loc[mtx[i]!=0][i].mean() # average rating for item i
    return(rating)


def predictTopRating(mtx, sim_mtx, k, n, user_x):
    lst = []
    for item in mtx.loc[user_x].loc[mtx.loc[user_x]==0].index.tolist(): #items no rankeados
        lst.append([item, predictItemRating(mtx, sim_mtx, k, user_x, item)])
    predictions = pd.DataFrame(lst, columns=['movieId', 'rating'])
    predictions = predictions.sort_values(by='rating', ascending=False).iloc[:n].set_index('movieId')
    movies = loadMovies()
    predictions['title'] = movies.loc[predictions.index.tolist()]['title'] 
    print(predictions)
    return(predictions)



k = 6
user_x = 1
item_i = 10
n = 10

rating_mtx, means = getRatingMatrix(loadRatings())
similarity_mtx = getSimilarityMatrix(rating_mtx)

predictions = predictTopRating(rating_mtx, similarity_mtx, k, n, user_x)

