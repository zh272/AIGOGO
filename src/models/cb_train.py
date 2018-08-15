
import os
import numpy as np
from tabulate import tabulate
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# import warnings
# warnings.simplefilter("ignore", UserWarning)

def get_submission(X, y, X_test, params, cate_col=None, col_types=None):

    
    categorical_features_indices = np.where(X.dtypes != np.float)[0]
    X.fillna(-999, inplace=True)
    X_test.fillna(-999,inplace=True)

    # create model
    model=CatBoostRegressor(**params)

    # train with validation
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=1234)
    model.fit(
        X_train, y_train, cat_features=categorical_features_indices,
        eval_set=(X_validation, y_validation), plot=False, early_stopping_rounds=20
    )

    # # train without validation
    # model.fit(
    #     X, y, cat_features=categorical_features_indices, plot=False
    # )
    
    print('====== CatBoost Feature Importances ======')
    feature_importances = np.array(model.feature_importances_)
    feature_names = X.columns.values
    sorted_idx = np.argsort(feature_importances*-1) # descending order
    for idx in sorted_idx:
        print('[{:20s}]'.format(feature_names[idx]), '{:7.4f}'.format(feature_importances[idx]))
    
    with open('summary.txt', 'w') as f:
        f.write('====== CatBoost Feature Importances ======\n')
        for idx in sorted_idx:
            f.write('[{:20s}] {:7.4f}'.format(feature_names[idx], feature_importances[idx]))
        f.write('\n')

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

    params = {
        'n_estimators':100000, 'learning_rate':20, 'objective':'MAE', 
        'max_depth':4, 'colsample_bylevel':0.7, 'reg_lambda':None, 'task_type': 'CPU'
    }

    model_output = get_submission(X_train, y_train, X_test, params)
    write_precessed_data(model_output['submission'])

