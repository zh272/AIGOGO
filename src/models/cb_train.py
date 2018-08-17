
import os
import numpy as np
from tabulate import tabulate
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# import warnings
# warnings.simplefilter("ignore", UserWarning)

# def get_submission(X, y, X_test, params, cate_col=None, col_types=None):

    
#     categorical_features_indices = np.where(X.dtypes != np.float)[0]
#     X.fillna(-999, inplace=True)
#     X_test.fillna(-999,inplace=True)

#     # create model
#     model=CatBoostRegressor(**params)

#     # train with validation
#     X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=1234)

def get_submission(X_train, y_train, X_validation, y_validation, X_test, params):
    categorical_features_indices = np.where(X_train.dtypes != np.float)[0]
    X_train.fillna(-999, inplace=True)
    X_validation.fillna(-999, inplace=True)
    X_test.fillna(-999,inplace=True)
    model=CatBoostRegressor(**params)



    model.fit(
        X_train, y_train, cat_features=categorical_features_indices,
        eval_set=(X_validation, y_validation), plot=False, early_stopping_rounds=None
    )

    # # train without validation
    # model.fit(
    #     X, y, cat_features=categorical_features_indices, plot=False
    # )
    
    print('====== CatBoost Feature Importances ======')
    feature_importances = np.array(model.feature_importances_)
    feature_names = X_train.columns.values
    sorted_idx = np.argsort(feature_importances*-1) # descending order
    for idx in sorted_idx:
        print('[{:20s}]'.format(feature_names[idx]), '{:7.4f}'.format(feature_importances[idx]))
    
    with open('summary.txt', 'w') as f:
        f.write('====== CatBoost Feature Importances ======\n')
        for idx in sorted_idx:
            f.write('[{:20s}] {:7.4f}\n'.format(feature_names[idx], feature_importances[idx]))
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

    # X_train = read_interim_data('X_train_prefs.csv')
    # X_test = read_interim_data('X_test_prefs.csv')
    # y_train = read_interim_data('y_train_prefs.csv')
    X_train = read_interim_data('X_train_prefs.csv')
    y_train = read_interim_data('y_train_prefs.csv')
    X_valid = read_interim_data('X_valid_prefs.csv')
    y_valid = read_interim_data('y_valid_prefs.csv')
    X_test = read_interim_data('X_test_prefs.csv')


    feature_list = ['cat_age', 'cat_area', 'cat_assured', 'cat_cancel', 'cat_distr', 'cat_marriage', 'cat_sex', 
    'cat_vc', 'cat_vmm1', 'cat_vmm2', 'cat_vmy', 'cat_vqpt', 'cat_vregion', 'cat_zip', 
    'int_acc_lia', 'int_claim', 'int_others', 'real_acc_dmg', 'real_acc_lia', 'real_loss', 
    'real_prem_dmg', 'real_prem_ins', 'real_prem_lia', 'real_prem_plc', 'real_prem_thf', 
    'real_prem_vc', 'real_vcost', 'real_ved', 'cat_sex_marr', 'real_freq_distr', 
    'real_prem_area_distr', 'real_prem_ic_distr', 'real_prem_distr', 'cat_vdom', 'cat_ved', 
    'cat_vmotor', 'cat_vmm1_vmy', 'real_prem_ved', 'real_prem_vmm1', 'real_prem_vmm2', 
    'real_prem_vmy', 'cat_claim_ins', 'real_loss_ins', 'real_prem_ic_nmf_1', 'real_prem_ic_nmf_2', 
    'real_prem_ic_nmf_3', 'real_prem_ic_nmf_4', 'real_prem_ic_nmf_5', 'real_prem_ic_nmf_6', 
    'real_prem_ic_nmf_7', 'real_mc_prob_distr']

    X_train = X_train[feature_list]
    X_valid = X_valid[feature_list]
    X_test = X_test[feature_list]

    params = {
        'n_estimators':50000, 'learning_rate':20, 'objective':'MAE', 
        'max_depth':4, 'colsample_bylevel':0.7, 'reg_lambda':None, 'task_type': 'CPU'
    }

    model_output = get_submission(X_train, y_train, X_valid, y_valid, X_test, params)
    write_precessed_data(model_output['submission'])

