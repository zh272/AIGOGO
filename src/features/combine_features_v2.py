from create_features_v2 import *
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import fire
import lightgbm as lgb
import pandas as pd
import sys
sys.path.insert(0, '../data')
from utils import read_data, write_data

'''
base case: 1799

+ premium by main coverage grp: 1799
+ premium on 0 insured amounts: 1799
+ premium avg of insured coverage: 1800
+ premium avg of insured coverage by vmy: 1799

+ real_acc_dmg: 1787
+ real_acc_lia: 1795
+ real_acc_dmg & lia: 1784

<<<<<<< HEAD
+ real_age: 1793
+ real_age_grp: 1794
+ real_age_tail: 1793

+ real_mc_mean_sex: 1790
+ real_mc_prob_sex: 1789

+ real_mc_mean_marriage: 1798
+ real_mc_prob_marriage: 1799

+ real_mc_mean_sex_marriage: 1791

+ real_mc_prob_sex & mean_marriage: 1791
+ real_mc_prob_sex & prob_marriage: 1791

+ real_mc_sex + real_age: 1787

+ real_mc_mean_zip: 1801
+ real_mc_prob_zip: 1803
=======
+ real_mc_mean_diff_dmg:

+ real_age: 1793
>>>>>>> dddb05ba206ebfa8eda1a37adff490ad5c444440

+ replacement cost: 1795
'''

######## get pre feature selection data set ########
def get_bs2_combined_features(df_policy, df_claim):
    '''
    In:
        DataFrame(df_policy),
        DataFrame(df_claim),

    Out:
        DataFrame(X_fs),
        DataFrame(y_fs),

    Description:
        create train dataset with additional columns
    '''
    print('Getting labels')
    y_train_all = read_data('training-set.csv', path='raw')
    y_test = read_data('testing-set.csv', path='raw')

    print('\nSplitting train valid label\n')
    y_train, y_valid = train_test_split(y_train_all, test_size=0.2, random_state=0)

    print('Getting neural network processed premiums')
    X_fs = read_data('premium_60_1.csv')

<<<<<<< HEAD
=======
    print('Getting column real_age')
    X_fs = X_fs.assign(real_age = get_bs2_real_age(df_policy, X_fs.index))

>>>>>>> dddb05ba206ebfa8eda1a37adff490ad5c444440
    print('\nSplitting train valid test features\n')
    X_train_all = X_fs.loc[y_train_all.index]
    X_train = X_fs.loc[y_train.index]
    X_valid = X_fs.loc[y_valid.index]
    X_test = X_fs.loc[y_test.index]

    print('Writing results to file')
    write_data(X_train, "X_train_bs2.csv")
    write_data(y_train, "y_train_bs2.csv")
    write_data(X_valid, "X_valid_bs2.csv")
    write_data(y_valid, "y_valid_bs2.csv")
    write_data(X_train_all, "X_train_all_bs2.csv")
    write_data(y_train_all, "y_train_all_bs2.csv")
    write_data(X_test, "X_test_bs2.csv")
    write_data(y_test, "y_test_bs2.csv")

    return(None)


<<<<<<< HEAD
def get_bs2_quick_mae(params, get_imp=True):
=======
def get_bs2_quick_mae(params):
>>>>>>> dddb05ba206ebfa8eda1a37adff490ad5c444440
    '''
    In:

    Out:
        float(mae)

    Description:
        calculate quick mae on validation set
    '''
    X_train = read_data('X_train_bs2.csv')
    y_train = read_data('y_train_bs2.csv')
    X_valid = read_data('X_valid_bs2.csv')
    y_valid = read_data('y_valid_bs2.csv')

    cols_train = [col for col in X_train.columns if not col.startswith('cat')]

    lgb_train = lgb.Dataset(X_train[cols_train].values, y_train.values.flatten(), free_raw_data=False)
    lgb_valid = lgb.Dataset(X_valid[cols_train].values, y_valid.values.flatten(), reference=lgb_train, free_raw_data=False)

    model = lgb.train(
        params['model'], lgb_train, valid_sets=lgb_valid, **params['train']
    )

    valid_pred = model.predict(X_valid[cols_train])
    valid_pred = pd.DataFrame(valid_pred, index = X_valid.index, columns = ['Next_Premium'])
    valid_mae = mean_absolute_error(y_valid, valid_pred)
    print('pre-selection mae is {}'.format(valid_mae))

<<<<<<< HEAD
    if get_imp:
        varimp = list(model.feature_importance())
        varimp = dict(zip(cols_train, varimp))
        for key, value in sorted(varimp.items(), key=lambda x: -x[1]):
            print("'%s': %s," % (key, value))
=======
    varimp = list(model.feature_importance())
    varimp = dict(zip(cols_train, varimp))
    for key, value in sorted(varimp.items(), key=lambda x: -x[1]):
        print("'%s': %s," % (key, value))
>>>>>>> dddb05ba206ebfa8eda1a37adff490ad5c444440

    return(None)


def demo():
#    df_claim = read_data('claim_0702.csv', path='raw')
#    df_policy = read_data('policy_0702.csv', path='raw')

    lgb_model_params = {
        'boosting_type': 'gbdt',
        'num_iterations': 5000,
        'objective': 'regression_l1',
        'metric': 'mae',
        'seed': 0,
    }
    lgb_train_params = {
        'early_stopping_rounds': 3,
        'learning_rates': lambda iter: max(0.1*(0.99**iter), 0.005)
    }
    lgb_params = {'model': lgb_model_params, 'train': lgb_train_params}

    get_bs2_combined_features(df_policy, df_claim)
    get_bs2_quick_mae(lgb_params)

if __name__ == '__main__':
    fire.Fire(demo)