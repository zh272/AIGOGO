# install on windows, see:
# https://www.ibm.com/developerworks/community/blogs/jfp/entry/Installing_XGBoost_For_Anaconda_on_Windows?lang=en

# import os
# mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-8.1.0-posix-seh-rt_v6-rev0\\mingw64\\bin'
# os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

# import pickle
import xgboost as xgb

import numpy as np
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error
from sklearn.datasets import load_iris, load_digits, load_boston

def xgb_trainer(X_train, y_train, X_test, y_test):

    # >>> Default params:
    # max_depth=3, learning_rate=0.1, n_estimators=100, silent=True, objective='reg:linear', 
    # booster='gbtree', n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, 
    # subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, 
    # scale_pos_weight=1, base_score=0.5, random_state=0, seed=None, missing=None

    # parameter reference: 
    # https://xgboost.readthedocs.io/en/latest/python/python_api.html?highlight=xgbregresso#xgboost.XGBRegressor

    # objective reference: 
    # https://xgboost.readthedocs.io/en/latest/parameter.html
        
    params = {'n_estimators': 50, 'learning_rate': 0.01, 'objective':'reg:tweedie',
        'max_delta_step':10, 'subsample':0.8, 'colsample_bytree':0.8}

    regressor = xgb.XGBRegressor(**params)
    regressor.fit(X_train, y_train, eval_metric='mae')
    # xxx=regressor.feature_importances_

    train_pred = regressor.predict(X_train)
    test_pred = regressor.predict(X_test)
    print('Training stats:')
    print(mean_absolute_error(y_train, train_pred))
    print('Testing stats:')
    print(mean_absolute_error(y_test, test_pred))
    # acc = regressor.score(X_test, y_test)

    return train_pred, test_pred

if __name__ == '__main__':
    print("Boston Housing: regression")
    boston = load_boston()
    y = boston['target']
    X = boston['data']
    rng = np.random.RandomState(31337)
    kf = KFold(n_splits=2, shuffle=True, random_state=rng)
    for train_index, test_index in kf.split(X):
        xgb_trainer(X[train_index], y[train_index], X[test_index], y[test_index])


















# print("Zeros and Ones from the Digits dataset: binary classification")
# digits = load_digits(2)
# y = digits['target']
# X = digits['data']
# rng = np.random.RandomState(31337)
# kf = KFold(n_splits=2, shuffle=True, random_state=rng)
# for train_index, test_index in kf.split(X):
#     xgb_model = xgb.XGBClassifier().fit(X[train_index], y[train_index])
#     predictions = xgb_model.predict(X[test_index])
#     actuals = y[test_index]
#     print(confusion_matrix(actuals, predictions))

# print("Boston Housing: regression")
# boston = load_boston()
# y = boston['target']
# X = boston['data']
# kf = KFold(n_splits=2, shuffle=True, random_state=rng)
# for train_index, test_index in kf.split(X):
#     xgb_model = xgb.XGBRegressor().fit(X[train_index], y[train_index])
#     predictions = xgb_model.predict(X[test_index])
#     actuals = y[test_index]
#     print(mean_squared_error(actuals, predictions))

# print("Parameter optimization")
# y = boston['target']
# X = boston['data']
# xgb_model = xgb.XGBRegressor()
# clf = GridSearchCV(xgb_model,
#                 {'max_depth': [2,4,6],
#                     'n_estimators': [50,100,200]}, verbose=1)
# clf.fit(X,y)
# print(clf.best_score_)
# print(clf.best_params_)

# # The sklearn API models are picklable
# print("Pickling sklearn API models")
# # must open in binary format to pickle
# pickle.dump(clf, open("best_boston.pkl", "wb"))
# clf2 = pickle.load(open("best_boston.pkl", "rb"))
# print(np.allclose(clf.predict(X), clf2.predict(X)))

# # Early-stopping

# X = digits['data']
# y = digits['target']
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# clf = xgb.XGBClassifier()
# clf.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="auc",
#         eval_set=[(X_test, y_test)])



















# import random, math, csv, metrics
# import numpy as np
# import xgboost as xgb
# from xgboost import XGBClassifier
# from metrics import objectivefn
# # from xgboost import XGBClassifier
# # from sklearn import xgboost
# from sklearn.preprocessing import normalize
# # from imblearn import metrics
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.over_sampling import ADASYN
# from sklearn.decomposition import PCA
# # from sklearn import ensemble


# ######### Set Parameters ###########
# fix_random = True
# if fix_random:
#     random.seed(29)
#     np.random.seed(29)
# do_normalize = False
# selected_feature =  range(55)
#                     # [0,1,2,5,7,9,17,18,19,20,21,25]
#                     # [12,13,14,16,18,24,28,31,34, ]
#                     # 11,17,21,22,25, ]
#                     # 36,37,43,44,45,46,47,49,50,54]  
#                     # range(55)
# do_pca = False
# n_pca = 8
# do_sample = True
# sampler_method = 'under' # 'under' or 'over'
# neg_label = 0
# ######### Load Data ###########
# X = []
# Y = []

# with open('data.csv', 'rb') as csvfile:
#     datareader = csv.reader(csvfile, delimiter=',' )
#     for row in datareader:
#         X.append([float(i) for i in row[0:-1]])
#         Y.append(float(row[-1]))

# X_array = np.asarray(X)
# Y_array = np.asarray(Y)
# X_array[X_array<0] = 0
# Y_array[Y_array==0] = neg_label

# split = 35000
# X_train = X_array[0:split,selected_feature]
# Y_train = Y_array[0:split]
# X_test = X_array[split:, selected_feature]
# Y_test = Y_array[split:]

# if do_normalize:
#     X_train = normalize(X_train, axis=1)
#     X_test = normalize(X_test, axis=1)


# ######### Perform PCA ###########
# if do_pca:
#     pca = PCA(n_components=n_pca)
#     X_pca = pca.fit_transform(X_train)
#     Y_pca = Y_train
# else:
#     X_pca = X_train
#     Y_pca = Y_train

# data_set = [X_train, Y_train]

# ########## Perform Re-sampling ###########
# if do_sample:
#     if sampler_method == 'under':
#         # sampler = RandomUnderSampler(ratio='auto', random_state=42)
#         num_sample = int( math.ceil(1.0*sum(Y_train==1)) )
#         # num_class0=len(Y_train)-num_class1
#         if fix_random:
#             sampler = RandomUnderSampler(ratio={neg_label:num_sample, 1:num_sample}, random_state=42)
#         else:
#             sampler = RandomUnderSampler(ratio={neg_label:num_sample, 1:num_sample})
#     elif sampler_method == 'over':
#         if fix_random:
#             sampler = ADASYN(random_state=42)
#         else:
#             sampler = ADASYN()
#     X_res, Y_res = sampler.fit_sample(X_pca, Y_pca)
# else:
#     X_res = X_pca
#     Y_res = Y_pca
# num_pos = sum(Y_res==1)
# num_neg = len(Y_res)-num_pos

# if fix_random:
#     params = {'n_estimators': 50, 'learning_rate': 0.01, 'random_state': 3, 'objective':objectivefn,
#             'scale_pos_weight':num_neg/num_pos, 'max_delta_step':10, 'subsample':1.0, 'colsample_bytree':1.0}
# else:
#     params = {'n_estimators': 50, 'learning_rate': 0.01, 'objective':'binary:logistic',
#             'scale_pos_weight':num_neg/num_pos, 'max_delta_step':10, 'subsample':0.8, 'colsample_bytree':0.8}

# clf = XGBClassifier(**params)
# clf.fit(X_res, Y_res, eval_metric='error')
# xxx=clf.feature_importances_
# # acc = clf.score(X_test, Y_test)
# y_train_pred0 = clf.predict(X_res)
# class_stat_train0 = metrics.class_stats(Y_res, y_train_pred0)

# y_train_pred = clf.predict(X_res)
# y_test_pred = clf.predict(X_test)

# print "real class1: %d" %sum(Y_res==1)
# print "predicted class1: %d" %sum(y_train_pred==1) 
# recall_train = metrics.geo_mean_recall(Y_train, y_train_pred)
# precision_train = metrics.geo_mean_precision(Y_train, y_train_pred)
# class_stat_train = metrics.class_stats(Y_train, y_train_pred)
# geo_test = metrics.geo_mean_recall(Y_test, y_test_pred)
# precision_test = metrics.geo_mean_precision(Y_test, y_test_pred)
# class_stat_test = metrics.class_stats(Y_test, y_test_pred)
# print("===== BEST INDIVIDUAL =====")
# print("train_TP=%d, train_FP=%d, train_TN=%d, train_FN=%d"%(class_stat_train[0],class_stat_train[1],class_stat_train[2],class_stat_train[3]))
# print("training geo_mean_recall: %4.3f, training geo_mean_precision: %4.3f"%(recall_train, precision_train) )
# print "train_accuracy=%4.3f"%(float(class_stat_train[0]+class_stat_train[2])/len(Y_train))
# print("test_TP=%d, test_FP=%d, test_TN=%d, test_FN=%d"%(class_stat_test[0],class_stat_test[1],class_stat_test[2],class_stat_test[3]))
# print("testing geo_mean_recall: %4.3f, testing geo_mean_precision: %4.3f"%(geo_test, precision_test) )
# print "test_accuracy=%4.3f"%(float(class_stat_test[0]+class_stat_test[2])/len(Y_test))


















# import xgboost as xgb
# # read in data
# dtrain = xgb.DMatrix('demo/data/agaricus.txt.train')
# dtest = xgb.DMatrix('demo/data/agaricus.txt.test')
# # specify parameters via map
# param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
# num_round = 2
# bst = xgb.train(param, dtrain, num_round)
# # make prediction
# preds = bst.predict(dtest)

