import numpy as np
import pandas as pd
import os
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import itertools

######## feature selection metrics ########
def get_lgb_mae(cols_train, train, valid, params):
    '''
    In:
        list(cols_train),
        DataFrame(train),
        DataFrame(valid),
        dict(params),

    Out:
        float(mae)

    Description:
        get valid dataset mae from lightgbm
    '''
    lgb_train = lgb.Dataset(train[cols_train].values, train['Next_Premium'].values.flatten(), free_raw_data=False)
    lgb_valid = lgb.Dataset(valid[cols_train].values, valid['Next_Premium'].values.flatten(), reference=lgb_train, free_raw_data=False)
    model = lgb.train(
        params['model'], lgb_train, valid_sets=lgb_valid, **params['train']
    )
    valid_pred = model.predict(valid[cols_train])
    valid_mae = mean_absolute_error(valid['Next_Premium'], valid_pred)
    varimp = list(model.feature_importance())

    train_pred = model.predict(train[cols_train])
    train_mae = mean_absolute_error(train['Next_Premium'], train_pred)
    print('Train mae is {}'.format(train_mae))

    cols_wst = []
    for i in range(min(20, len(varimp) - 1)):
        idx_min = varimp.index(min(varimp))
        varimp[idx_min] = max(varimp) + 1
        cols_wst.append(cols_train[idx_min])

    return(valid_mae, cols_wst)


def get_cat_mae(cols_train, train, valid, params):
    '''
    In:
        list(cols_train),
        DataFrame(train),
        DataFrame(valid),
        dict(params),

    Out:
        float(mae)

    Description:
        get valid dataset mae from catboost
    '''
    # get data for training
    X_train = train[cols_train]
    y_train = train['Next_Premium']
    X_valid = valid[cols_train]
    y_valid = valid['Next_Premium']

    # get categorical indices
    categorical_features_indices = np.where(X_train.dtypes != np.float)[0]

    # get random forest model
    model = CatBoostRegressor(**params)

    # tranform to h2o dataset
    model.fit(X_train,
              y_train,
              cat_features = categorical_features_indices,
              eval_set=(X_valid, y_valid),
              plot=False,
              early_stopping_rounds=2
              )

    mae = np.mean(abs(y_valid - model.get_test_eval()))
    varimp = model.get_feature_importance()
    cols_wst = []

    for i in range(5):
        idx_min = varimp.index(min(varimp))
        varimp[idx_min] = max(varimp) + 1
        cols_wst.append(cols_train[idx_min])

    return(mae, cols_wst)
