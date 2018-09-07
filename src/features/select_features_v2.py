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
    print('Getting column cat_ins')
    X_fs = X_fs.assign(cat_ins = get_bs2_cat(df_policy, X_fs.index, "Insured's_ID"))

    print('Getting column real_other')
    X_fs = X_fs.assign(real_other = get_bs2_cat(df_policy, X_fs.index, 'Multiple_Products_with_TmNewa_(Yes_or_No?)'))

    print('Getting column cat_assured')
    X_fs = X_fs.assign(cat_assured = get_bs2_cat(df_policy, X_fs.index, 'fassured'))

    print('Getting column real_prem_ic_assured')
    X_fs = X_fs.assign(real_assured_grp = X_fs['cat_assured'].map(lambda x: x % 2))
    X_fs = X_fs.assign(real_prem_ic_assured = get_bs2_real_prem_ic(df_policy, X_fs.index, X_fs['real_assured_grp']))

    print('Getting column real_age')
    X_fs = X_fs.assign(real_age = get_bs2_real_age(df_policy, X_fs.index))

    print('Getting column real_prem_ic_age')
    X_fs = X_fs.assign(real_age_grp = get_bs2_real_age_grp(df_policy, X_fs.index))
    X_fs = X_fs.assign(real_prem_ic_age = get_bs2_real_prem_ic(df_policy, X_fs.index, X_fs['real_age_grp']))

    print('Getting column cat_sex')
    X_fs = X_fs.assign(cat_sex = get_bs2_cat(df_policy, X_fs.index, 'fsex'))

    print('Getting column real_prem_ic_sex')
    X_fs = X_fs.assign(real_prem_ic_sex = get_bs2_real_prem_ic(df_policy, X_fs.index, X_fs['cat_sex']))

    print('Getting column cat_marriage')
    X_fs = X_fs.assign(cat_marriage = get_bs2_cat(df_policy, X_fs.index, 'fmarriage'))

    print('Getting column cat_zip')
    X_fs = X_fs.assign(cat_zip = get_bs2_cat(df_policy, X_fs.index, 'aassured_zip'))

    # policy
    print('Getting column real_cancel')
    X_fs = X_fs.assign(real_cancel = get_bs2_real_cancel(df_policy, X_fs.index))

    print('Getting column cat_ic_combo')
    X_fs = X_fs.assign(cat_ic_combo = get_bs2_cat_ic_combo(df_policy, X_fs.index))

    print('Getting column cat_ic_grp_combo')
    X_fs = X_fs.assign(cat_ic_grp_combo = get_bs2_cat_ic_grp_combo(df_policy, X_fs.index))

    print('Getting column cat_distr')
    X_fs = X_fs.assign(cat_distr = get_bs2_cat(df_policy, X_fs.index, 'Distribution_Channel'))

    print('Getting column real_acc_dmg')
    X_fs = X_fs.assign(real_acc_dmg = get_bs2_cat(df_policy, X_fs.index, 'pdmg_acc'))

    print('Getting column real_prem_ic_dmg')
    X_fs = X_fs.assign(real_prem_ic_dmg = get_bs2_real_prem_ic(df_policy, X_fs.index, X_fs['real_acc_dmg']))

    print('Getting column real_acc_lia')
    X_fs = X_fs.assign(real_acc_lia = get_bs2_cat(df_policy, X_fs.index, 'plia_acc'))

    print('Getting column real_prem_ic_lia')
    X_fs = X_fs.assign(real_prem_ic_lia = get_bs2_real_prem_ic(df_policy, X_fs.index, X_fs['real_acc_lia']))

    print('Getting column cat_area')
    X_fs = X_fs.assign(cat_area = get_bs2_cat(df_policy, X_fs.index, 'iply_area'))

    print('Getting column real_dage')
    X_fs = X_fs.assign(real_dage = get_bs2_real_dage(df_policy, X_fs.index))

    print('Getting column real_age')
    X_fs = X_fs.assign(real_age = get_bs2_real_age(df_policy, X_fs.index))

    print('Getting column real_ins_self')
    X_fs = X_fs.assign(real_ins_self = np.where(X_fs['real_dage'] == X_fs['real_age'], 1, 0) )

    print('Getting column real_prem_terminate')
    X_fs = X_fs.assign(real_prem_terminate = get_bs2_real_prem_terminate(df_policy, X_fs.index))

    # vehicle
    print('Getting column cat_vmm1')
    X_fs = X_fs.assign(cat_vmm1 = get_bs2_cat(df_policy, X_fs.index, 'Vehicle_Make_and_Model1'))

    print('Getting column cat_vmm2')
    X_fs = X_fs.assign(cat_vmm2 = get_bs2_cat(df_policy, X_fs.index, 'Vehicle_Make_and_Model2'))

    print('Getting column real_vmy')
    X_fs = X_fs.assign(real_vmy = get_bs2_real_vmy(df_policy, X_fs.index))

    print('Getting column real_vmy_tail')
    X_fs = X_fs.assign(real_vmy_tail = get_bs2_real_vmy_tail(df_policy, X_fs.index))

    print('Getting column real_prem_ic_vmy')
    X_fs = X_fs.assign(real_prem_ic_vmy = get_bs2_real_prem_ic(df_policy, X_fs.index, X_fs['real_vmy_tail']))

    print('Getting column real_vengine')
    X_fs = X_fs.assign(real_vengine = get_bs2_cat(df_policy, X_fs.index, 'Engine_Displacement_(Cubic_Centimeter)'))

    print('Getting column real_vengine_grp')
    X_fs = X_fs.assign(real_vengine_grp = get_bs2_real_vengine_grp(df_policy, X_fs.index))

    print('Getting column cat_vregion')
    X_fs = X_fs.assign(cat_vregion = get_bs2_cat(df_policy, X_fs.index, 'Imported_or_Domestic_Car'))

    print('Getting column cat_vc')
    X_fs = X_fs.assign(cat_vc = get_bs2_cat(df_policy, X_fs.index, 'Coding_of_Vehicle_Branding_&_Type'))

    print('Getting column real_vqpt')
    X_fs = X_fs.assign(real_vqpt = get_bs2_cat(df_policy, X_fs.index, 'qpt'))

    print('Getting column real_vcost')
    X_fs = X_fs.assign(real_vcost = get_bs2_cat(df_policy, X_fs.index, 'Replacement_cost_of_insured_vehicle'))

    print('Getting column real_vequip')
    X_fs = X_fs.assign(real_vequip = get_bs2_real_vequip(df_policy, X_fs.index))

    # claim
    df_claim16 = df_claim[df_claim['Accident_Date'].map(lambda x: str(x).startswith('2016'))]

    print('Getting column real_num_claim')
    X_fs = X_fs.assign(real_num_claim = get_bs2_real_num_claim(df_claim, X_fs.index))

    print('Getting column real_num_claim16')
    X_fs = X_fs.assign(real_num_claim16 = get_bs2_real_num_claim(df_claim16, X_fs.index))

    print('Getting column real_claim_self')
    X_fs = X_fs.assign(real_claim_self = get_bs2_real_claim_self(df_policy, df_claim, X_fs.index))

    print('Getting column real_claim_self16')
    X_fs = X_fs.assign(real_claim_self16 = get_bs2_real_claim_self(df_policy, df_claim16, X_fs.index))

    print('Getting column real_nearest_claim')
    X_fs = X_fs.assign(real_nearest_claim = get_bs2_real_nearest_claim(df_claim, X_fs.index))

    print('Getting column cat_claim_cause')
    X_fs = X_fs.assign(cat_claim_cause = get_bs2_cat_claim_cause(df_claim, X_fs.index))

    print('Getting column cat_claim_cause16')
    X_fs = X_fs.assign(cat_claim_cause16 = get_bs2_cat_claim_cause(df_claim16, X_fs.index))

    print('Getting column real_loss')
    X_fs = X_fs.assign(real_loss = get_bs2_real_claim(df_claim, X_fs.index, 'Paid_Loss_Amount'))

    print('Getting column real_loss16')
    X_fs = X_fs.assign(real_loss16 = get_bs2_real_claim(df_claim16, X_fs.index, 'Paid_Loss_Amount'))

    print('Getting column real_loss_ins')
    X_fs = X_fs.assign(real_loss_ins = get_bs2_real_loss_ins(df_policy, df_claim, X_fs.index))

    print('Getting column real_loss_ins16')
    X_fs = X_fs.assign(real_loss_ins16 = get_bs2_real_loss_ins(df_policy, df_claim16, X_fs.index))

    print('Getting column real_expense')
    X_fs = X_fs.assign(real_expense = get_bs2_real_claim(df_claim, X_fs.index, 'paid_Expenses_Amount'))

    print('Getting column real_expense16')
    X_fs = X_fs.assign(real_expense16 = get_bs2_real_claim(df_claim16, X_fs.index, 'paid_Expenses_Amount'))

    print('Getting column real_salvage')
    X_fs = X_fs.assign(real_salvage = get_bs2_real_claim(df_claim, X_fs.index, 'Salvage_or_Subrogation?'))

    print('Getting column real_salvage16')
    X_fs = X_fs.assign(real_salvage16 = get_bs2_real_claim(df_claim16, X_fs.index, 'Salvage_or_Subrogation?'))

    print('Getting column real_claim_fault')
    X_fs = X_fs.assign(real_claim_fault = get_bs2_real_claim_fault(df_claim, X_fs.index))

    print('Getting column real_claim_fault16')
    X_fs = X_fs.assign(real_claim_fault16 = get_bs2_real_claim_fault(df_claim16, X_fs.index))

    print('Getting column cat_claim_area')
    X_fs = X_fs.assign(cat_claim_area = get_bs2_cat_claim_area(df_claim, X_fs.index))

    print('Getting column cat_claim_area16')
    X_fs = X_fs.assign(cat_claim_area16 = get_bs2_cat_claim_area(df_claim16, X_fs.index))

    print('Getting column real_claimants')
    X_fs = X_fs.assign(real_claimants = get_bs2_real_claimants(df_claim, X_fs.index))

    print('Getting column real_claimants16')
    X_fs = X_fs.assign(real_claimants16 = get_bs2_real_claimants(df_claim16, X_fs.index))

    # helper columns
    print('Getting column real_prem_plc, for mean encoding use')
    X_fs = X_fs.assign(real_prem_plc = get_bs2_real_prem_plc(df_policy, X_fs.index))

    # train valid split
    print('\nSplitting train valid test features\n')
    X_train = X_fs.loc[y_train.index]
    X_valid = X_fs.loc[y_valid.index]

    # mean encoding
    cols_cat = [col for col in X_fs.columns if col.startswith('cat')]

    # add mean encoding on mean of next_premium
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

    # add mean encoding on mean of dividend btw next_premium and premium
    for col_cat in cols_cat:
        col_mean = col_cat.replace('cat_', 'real_mc_mean_div_')
        print('Getting column ' + col_mean)
        X_valid[col_mean] = get_bs2_real_mc_mean_div(col_cat, X_train, y_train, X_valid=X_valid, train_only=False, fold=5, prior=1000)
        X_train[col_mean] = get_bs2_real_mc_mean_div(col_cat, X_train, y_train, X_valid=pd.DataFrame(), train_only=True, fold=5, prior=1000)

    # add median encoding on median of next_premium
    for col_cat in cols_cat:
        col_median = col_cat.replace('cat_', 'real_mc_median_')
        print('Getting column ' + col_median)
        X_valid[col_median] = get_bs2_real_mc_median(col_cat, X_train, y_train, X_valid=X_valid, train_only=False, fold=5, prior=1000)
        X_train[col_median] = get_bs2_real_mc_median(col_cat, X_train, y_train, X_valid=pd.DataFrame(), train_only=True, fold=5, prior=1000)

    # add median encoding on median of diff btw next_premium and premium
    for col_cat in cols_cat:
        col_median = col_cat.replace('cat_', 'real_mc_median_diff_')
        print('Getting column ' + col_median)
        X_valid[col_median] = get_bs2_real_mc_median_diff(col_cat, X_train, y_train, X_valid=X_valid, train_only=False, fold=5, prior=1000)
        X_train[col_median] = get_bs2_real_mc_median_diff(col_cat, X_train, y_train, X_valid=pd.DataFrame(), train_only=True, fold=5, prior=1000)

    # add median encoding on median of diff btw next_premium and premium
    for col_cat in cols_cat:
        col_median = col_cat.replace('cat_', 'real_mc_median_div_')
        print('Getting column ' + col_median)
        X_valid[col_median] = get_bs2_real_mc_median_div(col_cat, X_train, y_train, X_valid=X_valid, train_only=False, fold=5, prior=1000)
        X_train[col_median] = get_bs2_real_mc_median_div(col_cat, X_train, y_train, X_valid=pd.DataFrame(), train_only=True, fold=5, prior=1000)

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


def select_bs2_features(params, cols_fs, cols_default, max_len=-1):
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

    max_len = len(cols_fs) + 1 if max_len == -1 else max_len
    lst_cols_fs = [cols_default]
    for l in range(1, max_len):
        for lst in itertools.combinations(cols_fs, l):
            lst_cols_fs.append(cols_default + list(lst))

    for cols_fs in lst_cols_fs:
        get_bs2_quick_mae(params, get_imp=True, data=data, cols_fs=cols_fs)

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

        'num_leaves': 31,
        'max_depth': -1,
        'min_data_in_leaf': 20,
        'bagging_fraction': 1.0,
        'bagging_freq': 0,
        'feature_fraction': 1.0,
        'max_bin': 255,
        'min_data_in_bin': 3,
        'max_delta_step': 0.0,
        'lambda_l1': 0.0,
        'lambda_l2': 0.0,
        'min_gain_to_split': 0.0,
        'bin_construct_sample_cnt': 220000,

        'seed': 0,
        'bagging_seed': 3,
        'feature_fraction_seed': 2,
    }
    lgb_train_params = {
        'early_stopping_rounds': 3,
        'learning_rates': lambda iter: max(0.1*(0.99**iter), 0.005),
        'verbose_eval': True,
    }
    lgb_params = {'model': lgb_model_params, 'train': lgb_train_params}

    cols_fs = []
    cols_default = ['real_prem_ic_nn_1', 'real_mc_median_ins', 'real_mc_median_div_ins', 'real_mc_prob_ins', 'real_mc_median_assured', 'real_mc_median_diff_assured', 'real_age', 'real_mc_median_sex', 'real_mc_median_diff_sex','real_mc_prob_marriage', 'real_mc_median_div_ic_grp_combo', 'real_mc_median_distr', 'real_mc_median_div_distr', 'real_mc_median_div_area', 'real_acc_dmg', 'real_acc_lia', 'real_cancel', 'real_dage', 'real_prem_terminate', 'real_mc_median_ic_combo', 'real_mc_median_div_ic_combo', 'real_vmy', 'real_vcost', 'cat_vengine', 'real_vqpt', 'real_mc_median_div_vregion', 'real_mc_mean_diff_vmm1', 'real_mc_mean_diff_vmm2', 'real_mc_mean_diff_vc', 'real_loss', 'real_loss_ins', 'real_salvage', 'real_mc_mean_div_claim_cause', 'real_mc_prob_claim_cause', 'real_mc_prob_claim_area', 'real_nearest_claim', 'real_num_claim', 'real_claim_fault', 'real_claimants']
    max_len = -1
    select_bs2_features(lgb_params, cols_fs, cols_default, max_len=max_len)

if __name__ == '__main__':
    fire.Fire(demo)