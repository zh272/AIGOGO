
import os
import numpy as np
from tabulate import tabulate
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from helpers import MultiColumnLabelEncoder
# import warnings
# warnings.simplefilter("ignore", UserWarning)

def get_submission(X, y, X_test, model_params={}, train_params={}, cate_col=None, col_types=None):

    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=1234)

    # create dataset for lightgbm
    # if you want to re-use data, remember to set free_raw_data=False
    lgb_train = lgb.Dataset(X_train.values, y_train.values.flatten(),
                            free_raw_data=False)
    lgb_eval = lgb.Dataset(X_validation.values, y_validation.values.flatten(), reference=lgb_train,
                        free_raw_data=False)

    # specify your configurations as a dict
    


    print('Start training...')
    model = lgb.train(
        model_params, lgb_train, valid_sets=lgb_eval, **train_params
    )


    feature_importances = np.array(model.feature_importance())
    feature_names = X.columns.values
    sorted_idx = np.argsort(feature_importances*-1) # descending order

    y_train_pred = model.predict(X_train)
    y_valid_pred =model.predict(X_validation)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    valid_mae = mean_absolute_error(y_validation, y_valid_pred)
    
    print('====== LightGBM Feature Importances ======')
    for idx in sorted_idx:
        print('[{:20s}] {:7.4f}'.format(feature_names[idx], feature_importances[idx]))
    print('training MAE:{}'.format(train_mae))
    print('validation MAE:{}'.format(valid_mae))
    
    with open('summary.txt', 'w') as f:
        f.write('====== LightGBM Feature Importances ======\n')
        for idx in sorted_idx:
            f.write('[{:20s}] {:7.4f}\n'.format(feature_names[idx], feature_importances[idx]))
        f.write('training MAE:{}\n'.format(train_mae))
        f.write('validation MAE:{}\n'.format(valid_mae))

    submission = pd.DataFrame()
    submission['Policy_Number'] = X_test.index
    submission['Next_Premium'] = model.predict(X_test)
    submission = submission.set_index(['Policy_Number'])

    return({'model': model, 'submission': submission})


def read_interim_data(file_name, index_col='Policy_Number'):
    '''
    In: file_name
    Out: interim_data
    Description: read data from directory /data/interim
    '''
    # set the path of raw data
    interim_data_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), os.path.pardir, os.path.pardir, 'data', 'interim'
    )

    file_path = os.path.join(interim_data_path, file_name)
    interim_data = pd.read_csv(file_path, index_col=index_col)

    # interim_data = interim_data.apply(lambda x:x.fillna(x.value_counts().index[0]))


    return(interim_data)

def write_precessed_data(df):
    '''
    In:
        DataFrame(df),
        str(file_name),
    Out:
        None
    Description:
        Write sample data to directory /data/interim
    '''
    precessed_data_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), os.path.pardir, os.path.pardir, 'data', 'processed'
    )
    write_sample_path = os.path.join(precessed_data_path, 'testing-set.csv')
    df.to_csv(write_sample_path)

    return(None)

if __name__ == '__main__':

    X_train = read_interim_data('X_train_bs.csv')
    X_test = read_interim_data('X_test_bs.csv')
    y_train = read_interim_data('y_train_bs.csv')

    categorical_features = []
    for name in X_train.columns.values:
        if 'cat' in name or 'int' in name:
            categorical_features.append(name)
            X_train[name] = X_train[name].astype('category')
            X_test[name] = X_test[name].astype('category')

    
    le = MultiColumnLabelEncoder(columns=categorical_features)
    le.fit(pd.concat([X_train, X_test]))
    X_train = le.transform(X_train)
    X_test = le.transform(X_test)
            

    model_params = {
        'boosting_type': 'gbdt',
        'num_iterations': 2000, 
        'max_depth':4,
        'max_bin':255,
        'objective': 'regression',
        'metric': 'mae',
        'reg_alpha':0.5,
        'num_leaves': 31,
        'learning_rate': 0.02,
        'colsample_bytree': 0.9,
        'subsample': 0.8,
        'subsample_freq': 5,
        'device': 'cpu'
    }
    train_params = {
        'early_stopping_rounds':None,
        'learning_rates': None, # lambda iter: 0.1*(0.99**iter),
        
    } 

    model_output = get_submission(X_train, y_train, X_test, model_params=model_params, train_params=train_params)
    write_precessed_data(model_output['submission'])
