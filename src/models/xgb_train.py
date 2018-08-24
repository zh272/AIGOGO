import os
import time
import fire
import random
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

## to detach from monitor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from helpers import test_epoch, ready, save_obj, load_obj

def get_submission(
    X_train, X_valid, y_train, y_valid, X_test, train_params={},
    get_train=False, save=False, load=False, mdl_name='xgb.pt'
):  
    
    PATH = './saved_model'
    if not os.path.isdir(PATH): os.makedirs(PATH)
    
    start_time = time.time()
    end_time = start_time
    if load:
        regressor = load_obj(mdl_name)
    else:
        regressor = xgb.XGBRegressor(**train_params)
        regressor.fit(X_train.values, y_train.values, eval_metric='mae')
        end_time = time.time()

        if save:
            save_obj(regressor, mdl_name)

    train_pred = regressor.predict(X_train.values)
    valid_pred = regressor.predict(X_valid.values)
    test_pred = regressor.predict(X_test.values)


            
    train_loss = mean_absolute_error(y_train.values, train_pred)
    valid_loss = mean_absolute_error(y_valid.values, valid_pred)

    feature_importances = regressor.feature_importances_
    
    feature_names = X_train.columns.values
    sorted_idx = np.argsort(feature_importances*-1) # descending order
    
    summary = '====== XGBoost Training Summary ======\n'
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

    return(None)

# empirical scale: weight_decay=0.0001
def demo(
    epochs=300, base_lr=0.05, max_depth=4, subsample=0.8, objective='reg:linear',
    colsample_bytree=0.8, colsample_bylevel=0.8, gamma=0.0, reg_alpha=3.0, reg_lambda=0.0,
    max_delta_step=0, get_train=False, save=False, load=False, seed=None
):
    if seed is not None:
        # known best seed=10
        rand_reset(seed)
    X_train = read_interim_data('X_train_prefs.csv')
    y_train = read_interim_data('y_train_prefs.csv')
    X_valid = read_interim_data('X_valid_prefs.csv')
    y_valid = read_interim_data('y_valid_prefs.csv')
    X_test = read_interim_data('X_test_prefs.csv')

    feature_list = [feature for feature in X_train.columns.values if 'cat_' not in feature]
    print('Number of features: {}'.format(len(feature_list)))

    # Filter features
    X_train = X_train[feature_list]
    X_valid = X_valid[feature_list]
    X_test = X_test[feature_list]

    ### Fill Missing Values
    X_train = X_train.apply(lambda x:x.fillna(-1))
    X_test = X_test.apply(lambda x:x.fillna(-1))

    # begin training
    train_params = {
        'n_estimators':epochs, 'learning_rate':base_lr, 'objective':'reg:linear',
        'max_delta_step':max_delta_step, 'max_depth':max_depth, 'subsample':subsample, 
        'colsample_bytree':colsample_bytree, 'colsample_bylevel':colsample_bylevel, 
        'gamma':gamma, 'reg_alpha':reg_alpha, 'reg_lambda':reg_lambda
    }
    
    model_output = get_submission(
        X_train, X_valid, y_train, y_valid, X_test, 
        train_params=train_params, get_train=get_train,
        save=save, load=load
    )

    summary = model_output['summary']
    summary += '>>> random seed: {}\n'.format(seed)

    print(summary)
    with open('summary_{}.txt'.format(int(model_output['valid_loss'])), 'w') as f:
        f.write(summary)

    # generate submission
    write_precessed_data(model_output['submission'], suffix='xgbtest{}'.format(int(model_output['valid_loss'])))
    if model_output['submission_train'] is not None:
        write_precessed_data(model_output['submission_train'], suffix='xgbtrain')
        write_precessed_data(model_output['submission_valid'], suffix='xgbvalid')

def rand_reset(seed):
    random.seed(seed)
    np.random.seed(random.randint(0,1000))

if __name__ == '__main__':
    # Example usage: "python nn_train.py --epochs 100"
    fire.Fire(demo)