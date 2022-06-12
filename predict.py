import numpy as np
import pandas as pd
from typing import List
from loadData import meltRatingMatrix

from baseline import BaselineItems, BaselineUsers
from ALS import ALS
from SVD import Mi_SVD

from surprise import SVD, Reader, Dataset

import tensorflow as tf
tf.get_logger().setLevel('ERROR') # only show error messages

from recommenders.utils.timer import Timer
from recommenders.models.ncf.ncf_singlenode import NCF


BATCH_SIZE, SEED = 256, 42

K_ALS, LAMBDA_ALS = 15, 0.05
K_SVD = 15
K_SVD_S, REG_SVD_S, EPOCHS_SVD_S = 100, 0.05, 100
FACTORS_GMF, EPOCHS_GMF = 512, 30
FACTORS_MLP, LAYERS_MLP, EPOCHS_MLP= 512, [16, 8, 4], 30
FACTORS_NEUMF, LAYERS_NEUMF, EPOCHS_NEUMF = 512, [16, 8, 4], 30


def getPredictions(train: pd.DataFrame, test: pd.DataFrame, models: List, data= None):
    # train: matriz de entrenamiento
    # test: matriz de test a predecir
    # models: lista de modelos a entrenar
    # data: si algun modelo usa NCF, se necesita el Dataset NCF

    model_options = ['Baseline_U', 'Baseline_U', 'ALS', 'SVD', 'SVD_S', 'GMF', 'MLP', 'NeuMF']
    # Matrix to dataframe
    train_melted = meltRatingMatrix(train).sort_values(by=['userID', 'itemID'])
    test_melted = meltRatingMatrix(test).sort_values(by=['userID', 'itemID'])

    pred_all = test_melted.copy()
    exec_time = pd.DataFrame({
        'model': models, 
        'train_time': np.NaN*len(models), 
        'predict_time':np.NaN*len(models)
    }).set_index('model')

    for name in models:
        assert name in model_options, "Modelo no válido"
        if name == 'Baseline_U':
            model = BaselineUsers()
            # train
            with Timer() as train_time:
                model.fit(train)
            # predict
            with Timer() as test_time:
                pred = model.test(test_melted[['userID', 'itemID']])
            exec_time.loc[name] = [train_time, test_time]
            
        elif name == 'Baseline_I':
            model = BaselineItems()
            # train
            with Timer() as train_time:
                model.fit(train)
            # predict
            with Timer() as test_time:
                pred = model.test(test_melted[['userID', 'itemID']])
            exec_time.loc[name] = [train_time, test_time]

        elif name == 'ALS':
            model = ALS()
            # train
            with Timer() as train_time:
                model.fit(train, K_ALS, LAMBDA_ALS)
            # predict
            with Timer() as test_time:
                pred = model.test(test_melted[['userID', 'itemID']])
            exec_time.loc[name] = [train_time, test_time]

        elif name == 'SVD':
            model = Mi_SVD()
            # train
            with Timer() as train_time:
                model.fit(train, K_SVD)
            # predict
            with Timer() as test_time:
                pred = model.test(test_melted[['userID', 'itemID']])
            exec_time.loc[name] = [train_time, test_time]

        elif name == 'SVD_S':
            # A reader is needed with the rating_scale param requiered
            reader = Reader(rating_scale=(0, 1))
            # The columns must correspond to user id, item id and ratings (in that order).
            trainset = Dataset.load_from_df(train_melted, reader)
            testset = Dataset.load_from_df(test_melted, reader)

            model = SVD(n_factors=K_SVD_S,\
                        n_epochs=EPOCHS_SVD_S,\
                        reg_all=REG_SVD_S)
            # train
            with Timer() as train_time:
                model.fit(trainset.build_full_trainset())
            # predict
            with Timer() as test_time:
                pred = model.test(testset.build_full_trainset().build_testset())
                # ordeno los resultados igual que test_melted y me quedo con la columna de estimados
                pred = list(pd.DataFrame(pred).sort_values(by=['uid', 'iid'])['est'])
            exec_time.loc[name] = [train_time, test_time]

        elif name == 'GMF':
            model = NCF (
                n_users=data.n_users, 
                n_items=data.n_items,
                model_type="GMF",
                n_factors=FACTORS_GMF,
                n_epochs=EPOCHS_GMF,
                batch_size=BATCH_SIZE,
                learning_rate=1e-3,
                verbose=10,
                seed=SEED
            )
            # train
            with Timer() as train_time:
                model.fit(data)
            # predict
            with Timer() as test_time:
                pred = model.predict(test_melted.userID, test_melted.itemID, is_list=True)
            exec_time.loc[name] = [train_time, test_time]

        elif name == 'MLP':
            model = NCF (
                n_users=data.n_users, 
                n_items=data.n_items,
                model_type="mlp",
                n_factors=FACTORS_MLP,
                layer_sizes=LAYERS_MLP,                
                n_epochs=EPOCHS_MLP,
                batch_size=BATCH_SIZE,
                learning_rate=1e-3,
                verbose=10,
                seed=SEED
            )
            # train
            with Timer() as train_time:
                model.fit(data)
            # predict
            with Timer() as test_time:
                pred = model.predict(test_melted.userID, test_melted.itemID, is_list=True)
            exec_time.loc[name] = [train_time, test_time]

        elif name == 'NeuMF':
            model = NCF (
                n_users=data.n_users, 
                n_items=data.n_items,
                model_type="neumf",
                n_factors=FACTORS_NEUMF,
                layer_sizes=LAYERS_NEUMF,                
                n_epochs=EPOCHS_NEUMF,
                batch_size=BATCH_SIZE,
                learning_rate=1e-3,
                verbose=10,
                seed=SEED
            )
            # train
            with Timer() as train_time:
                model.fit(data)
            # predict
            with Timer() as test_time:
                pred = model.predict(test_melted.userID, test_melted.itemID, is_list=True)
            exec_time.loc[name] = [train_time, test_time]

        pred_all[name] = pred
    
    return pred_all, exec_time


def _getRankings(ratings: pd.DataFrame, ind='userID', col='itemID', val='rating'):
    # orden descendente de rating, agrupacion por usuario y agregacion de itemID
    rankings = ratings.sort_values(by=val, ascending=False).\
                  groupby(ind).agg(ranking = (col, list)).\
                  apply(lambda x: np.asarray(x))
    return rankings


def getRankings(pred_all: pd.DataFrame):
    models = pred_all.columns[3:].tolist()
    rank_all = _getRankings(pred_all[['userID','itemID','rating']])
    for model in models:
        rank_all[model] = _getRankings(pred_all[['userID','itemID',model]], val=model)
        rank_all.insert(1, 'length', rank_all.ranking.apply(lambda r: len(r)))
    return rank_all
