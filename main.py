from pickle import TRUE
import numpy as np
import pandas as pd
import loadData as ld
from recommenders.models.ncf.dataset import Dataset as NCFDataset
from predict import getPredictions, getRankings
from evaluate import compareRMSE, compareRankings, getMetrics

SEED, BINARY_RATINGS = 42, False
np.random.seed(SEED)

# Cargar datos
r = ld.loadRatings()
# m = ld.getFilteredRatingMatrix(r, 30, 50)
m = ld.getFilteredRatingMatrix(r, 5, 0)

# Split Train Test
train, test = ld.splitRatings(m, 0.2)

# Normalization
train, norm_train = ld.normalizeRatings(train)
test, norm_test = ld.normalizeRatings(test)
n_m, norm_m = ld.normalizeRatings(m)

# Matrix to dataframe
train_melted = ld.meltRatingMatrix(train).sort_values(by=['userID', 'itemID'])
test_melted = ld.meltRatingMatrix(test).sort_values(by=['userID', 'itemID'])

print('Train elements: ', np.count_nonzero(~np.isnan(train)))
print('Test elements: ', np.count_nonzero(~np.isnan(test)))
print('Train shape: ', train.shape)
print('Test shape: ', test.shape)


##################   NCF DATASETS   ##################
# Write datasets to csv files.
train_file, test_file = "./train.csv", "./test.csv"
train_melted.to_csv(train_file, index=False)
test_melted.to_csv(test_file, index=False)
# Generate an NCF dataset object from the data subsets.
data = NCFDataset(train_file=train_file,
                  test_file=test_file,
                  seed=SEED,
                  binary=BINARY_RATINGS,
                  overwrite_test_file_full=True
                 )


##################   TRAIN AND PREDICT   ##################
models = ['Baseline_U', 'Baseline_U', 'ALS', 'SVD', 'SVD_S', 'GMF', 'MLP', 'NeuMF']
pred_all, exec_time = getPredictions(train, test, models, data)
print('Train and predict time: \n', exec_time)

rank_all = getRankings(pred_all)


##################   EVALUATE   ##################
compare_RMSE = compareRMSE(pred_all, plot=True)
print('RMSE: \n', compare_RMSE)

error_ranks = compareRankings(rank_all, plot=True)
#print('Comparación de rankings: \n', error_ranks)

metrics_results = getMetrics(rank_all, test_melted, plot=True)
