
import os
import fire
import time
import random
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error

from helpers import load_obj, save_obj
# import warnings
# warnings.simplefilter("ignore", UserWarning)

def get_submission(
    X_train, y_train, X_valid, y_valid, X_test, params,
    save=False, load=False, mdl_name='catb'
):
    categorical_features_indices = np.where(X_train.dtypes != np.float)[0]
    X_train.fillna(-999, inplace=True)
    X_valid.fillna(-999, inplace=True)
    X_test.fillna(-999,inplace=True)



    PATH = './saved_models'
    if not os.path.isdir(PATH): os.makedirs(PATH)
    
    start_time = time.time()
    end_time = start_time
    if load:
        regressor = load_obj(mdl_name)
    else:
        regressor=CatBoostRegressor(**params)

        regressor.fit(
            X_train, y_train, cat_features=categorical_features_indices,
            eval_set=(X_valid, y_valid), plot=False, early_stopping_rounds=None
        )
        end_time = time.time()

        if save:
            save_obj(regressor, mdl_name)

    train_pred = regressor.predict(X_train.values)
    valid_pred = regressor.predict(X_valid.values)
    test_pred = regressor.predict(X_test.values)
            
    train_loss = mean_absolute_error(y_train.values, train_pred)
    valid_loss = mean_absolute_error(y_valid.values, valid_pred)

    feature_importances = np.array(regressor.feature_importances_)
    
    feature_names = X_train.columns.values
    sorted_idx = np.argsort(feature_importances*-1) # descending order
    
    summary = '====== CatBoost Training Summary ======\n'
    for idx in sorted_idx:
        summary += '[{:<25s}] | {:<10.4f}\n'.format(feature_names[idx], feature_importances[idx])
    summary += '>>> training_time={:10.2f}min\n'.format((end_time-start_time)/60)
    summary += '>>> Final MAE: {:10.4f}(Training), {:10.4f}(Validation)\n'.format(train_loss,valid_loss)

    # Generate submission
    submission = pd.DataFrame(data=test_pred,index=X_test.index, columns=['Next_Premium'])

    submission_train = pd.DataFrame(data=train_pred,index=X_train.index, columns=['Next_Premium'])
    
    submission_valid = pd.DataFrame(data=valid_pred,index=X_valid.index, columns=['Next_Premium'])

    return {
        'model': regressor, 'submission': submission, 
        'submission_train':submission_train, 'submission_valid':submission_valid,
        'valid_loss':valid_loss, 'summary':summary
    }


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

def write_precessed_data(df, suffix=None):
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
    if suffix is None:
        file_name = 'testing-set.csv'
    else:
        file_name = 'testing-set_{}.csv'.format(suffix)
    write_sample_path = os.path.join(precessed_data_path, file_name)
    df.to_csv(write_sample_path)


def demo(
    epochs=20000, lr=50, objective='MAE', task_type='CPU',
    max_depth=4, colsample_bylevel=0.7, reg_lambda=None,
    get_train=False, get_test=True, save=False, load=False, seed=None
):
    if seed is not None:
        rand_reset(seed)

    if task_type=='GPU':
        colsample_bylevel=None
    
    X_train = read_interim_data('X_train_new.csv')
    y_train = read_interim_data('y_train_new.csv')
    X_valid = read_interim_data('X_valid_new.csv')
    y_valid = read_interim_data('y_valid_new.csv')
    X_test = read_interim_data('X_test_new.csv')


    feature_list = X_train.columns.values
    # feature_list = [feature for feature in X_train.columns.values if 'cat_' not in feature]
    num_features = len(feature_list)
    print('Number of features: {}'.format(num_features))

    X_train = X_train[feature_list]
    X_valid = X_valid[feature_list]
    X_test = X_test[feature_list]

    # colsample_bylevel should be None if task_type is 'GPU'
    params = {
        'n_estimators':epochs, 'learning_rate':lr, 'objective':objective, 
        'max_depth':max_depth, 'colsample_bylevel':None, 'reg_lambda':reg_lambda, 
        'task_type': task_type
    }

    model_output = get_submission(
        X_train, y_train, X_valid, y_valid, X_test, params,
        save=save, load=load, mdl_name='catb'
    )

    summary = model_output['summary']
    summary += '>>> random seed: {}\n'.format(seed)

    print(summary)
    with open('summary_cb{}.txt'.format(int(model_output['valid_loss'])), 'w') as f:
        f.write(summary)

    # generate submission
    if get_test:
        write_precessed_data(model_output['submission'], suffix='cbtest{}'.format(int(model_output['valid_loss'])))
    if get_train:
        write_precessed_data(model_output['submission_train'], suffix='cbtrain')
        write_precessed_data(model_output['submission_valid'], suffix='cbvalid')

def rand_reset(seed):
    random.seed(seed)
    np.random.seed(random.randint(0,1000))


if __name__ == '__main__':
    fire.Fire(demo)