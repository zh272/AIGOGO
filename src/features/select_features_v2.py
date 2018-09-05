from copy import deepcopy
from create_features_v2 import *
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import fire
import itertools
import lightgbm as lgb
import pandas as pd
import sys
sys.path.insert(0, '../data')
from utils import read_data, write_data

######## get pre feature selection data set ########
def get_bs2_feature_selection(df_policy, df_claim):
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

    print('\nSplitting train valid label\n')
    y_train, y_valid = train_test_split(y_train_all, test_size=0.2, random_state=0)

    print('Getting neural network processed premiums')
    X_fs = read_data('premium_60_1.csv')

    # insured
    print('Getting column real_age')
    X_fs = X_fs.assign(real_age = get_bs2_real_age(df_policy, X_fs.index))

    print('Getting column cat_sex')
    X_fs = X_fs.assign(cat_sex = get_bs2_cat(df_policy, X_fs.index, 'fsex'))

    print('Getting column cat_marriage')
    X_fs = X_fs.assign(cat_marriage = get_bs2_cat(df_policy, X_fs.index, 'fmarriage'))

    # policy
    print('Getting column real_acc_dmg')
    X_fs = X_fs.assign(real_acc_dmg = get_bs2_cat(df_policy, X_fs.index, 'pdmg_acc'))

    print('Getting column real_acc_lia')
    X_fs = X_fs.assign(real_acc_lia = get_bs2_cat(df_policy, X_fs.index, 'plia_acc'))

    print('Getting column real_prem_plc, for mean encoding use')
    X_fs = X_fs.assign(real_prem_plc = get_bs2_real_prem_plc(df_policy, X_fs.index))

    # train valid split
    print('\nSplitting train valid test features\n')
    X_train = X_fs.loc[y_train.index]
    X_valid = X_fs.loc[y_valid.index]

    # mean encoding
    cols_cat = [col for col in X_fs.columns if col.startswith('cat')]

    for col_cat in cols_cat:
        col_mean = col_cat.replace('cat_', 'real_mc_mean_')
        print('Getting column ' + col_mean)
        X_valid[col_mean] = get_bs2_real_mc_mean(col_cat, X_train, y_train, X_valid=X_valid, train_only=False, fold=5, prior=1000)
        X_train[col_mean] = get_bs2_real_mc_mean(col_cat, X_train, y_train, X_valid=pd.DataFrame(), train_only=True, fold=5, prior=1000)

    # add mean encoding on mean of diff btw next_premium and premium
    for col_cat in cols_cat:
        col_mean = col_cat.replace('cat_', 'real_mc_mean_diff_')
        print('Getting column ' + col_mean)
        X_valid[col_mean] = get_bs2_real_mc_mean_diff(col_cat, X_train, y_train, X_valid=X_valid, train_only=False, fold=5, prior=1000)
        X_train[col_mean] = get_bs2_real_mc_mean_diff(col_cat, X_train, y_train, X_valid=pd.DataFrame(), train_only=True, fold=5, prior=1000)

    # add mean encoding on probability of next_premium being 0
    for col_cat in cols_cat:
        col_prob = col_cat.replace('cat_', 'real_mc_prob_')
        print('Getting column ' + col_prob)
        X_valid[col_prob] = get_bs2_real_mc_prob(col_cat, X_train, y_train, X_valid=X_valid, train_only=False, fold=5, prior=1000)
        X_train[col_prob] = get_bs2_real_mc_prob(col_cat, X_train, y_train, X_valid=pd.DataFrame(), train_only=True, fold=5, prior=1000)

    print('Writing results to file')
    write_data(X_train, "X_train_fs.csv")
    write_data(y_train, "y_train_fs.csv")
    write_data(X_valid, "X_valid_fs.csv")
    write_data(y_valid, "y_valid_fs.csv")

    return(None)


def get_bs2_quick_mae(params, get_imp=False, data=[], cols_fs=[]):
    '''
    In:

    Out:
        float(mae)

    Description:
        calculate quick mae on validation set
    '''
    if len(data) != 0:
        X_train = data[0]
        y_train = data[1]
        X_valid = data[2]
        y_valid = data[3]
    else:
        X_train = read_data('X_train_fs.csv')
        y_train = read_data('y_train_fs.csv')
        X_valid = read_data('X_valid_fs.csv')
        y_valid = read_data('y_valid_fs.csv')

    cols_fs = cols_fs if len(cols_fs) != 0 else X_train.columns
    cols_train = [col for col in cols_fs if not col.startswith('cat')]

    lgb_train = lgb.Dataset(X_train[cols_train].values, y_train.values.flatten(), free_raw_data=False)
    lgb_valid = lgb.Dataset(X_valid[cols_train].values, y_valid.values.flatten(), reference=lgb_train, free_raw_data=False)

    lgb_params = deepcopy(params)
    model = lgb.train(
        lgb_params['model'], lgb_train, valid_sets=lgb_valid, **lgb_params['train']
    )

    valid_pred = model.predict(X_valid[cols_train])
    valid_pred = pd.DataFrame(valid_pred, index = X_valid.index, columns = ['Next_Premium'])
    valid_mae = mean_absolute_error(y_valid, valid_pred)
    print('{}: {}'.format(cols_train, valid_mae))

    if get_imp:
        varimp = list(model.feature_importance())
        varimp = dict(zip(cols_train, varimp))
        for key, value in sorted(varimp.items(), key=lambda x: -x[1]):
            print("'%s': %s," % (key, value))

    return(None)


def select_bs2_features(params, cols_fs, cols_default):
    '''
    In:
        list(cols_fs)
    Out:
        float(mae)

    Description:
        calculate quick mae on validation set
    '''
    X_train = read_data('X_train_fs.csv')
    y_train = read_data('y_train_fs.csv')
    X_valid = read_data('X_valid_fs.csv')
    y_valid = read_data('y_valid_fs.csv')
    data = [X_train, y_train, X_valid, y_valid]

    lst_cols_fs = []
    for l in range(1, len(cols_fs) + 1):
        for lst in itertools.combinations(cols_fs, l):
            lst_cols_fs.append(cols_default + list(lst))

    for cols_fs in lst_cols_fs:
        get_bs2_quick_mae(params, get_imp=False, data=data, cols_fs=cols_fs)

    return(None)


def demo():
#    df_claim = read_data('claim_0702.csv', path='raw')
#    df_policy = read_data('policy_0702.csv', path='raw')
#    get_bs2_feature_selection(df_policy, df_claim)

    lgb_model_params = {
        'boosting_type': 'gbdt',
        'num_iterations': 5000,
        'objective': 'regression_l1',
        'metric': 'mae',
        'seed': 0,
    }
    lgb_train_params = {
        'early_stopping_rounds': 3,
        'learning_rates': lambda iter: max(0.1*(0.99**iter), 0.005),
        'verbose_eval': False,
    }
    lgb_params = {'model': lgb_model_params, 'train': lgb_train_params}

    cols_fs = ['real_mc_mean_sex', 'real_mc_mean_diff_sex', 'real_mc_prob_sex']
    cols_default = ['real_prem_ic_nn_1', 'real_age', 'real_acc_dmg', 'real_acc_lia']
    select_bs2_features(lgb_params, cols_fs, cols_default)

if __name__ == '__main__':
    fire.Fire(demo)