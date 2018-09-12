from copy import deepcopy
from create_features_v2 import *
import fire
import lightgbm as lgb
import pandas as pd
import sys
sys.path.insert(0, '../data')
from utils import read_data, write_data

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

    print('Getting neural network processed premiums')
    X_fs = read_data('premium_60_1.csv')

    # insured
    print('Getting column cat_ins')
    X_fs = X_fs.assign(cat_ins = get_bs2_cat(df_policy, X_fs.index, "Insured's_ID"))

    print('Getting column cat_assured')
    X_fs = X_fs.assign(cat_assured = get_bs2_cat(df_policy, X_fs.index, 'fassured'))

    print('Getting column real_age')
    X_fs = X_fs.assign(real_age = get_bs2_real_age(df_policy, X_fs.index))

    print('Getting column cat_sex')
    X_fs = X_fs.assign(cat_sex = get_bs2_cat(df_policy, X_fs.index, 'fsex'))

    print('Getting column cat_marriage')
    X_fs = X_fs.assign(cat_marriage = get_bs2_cat(df_policy, X_fs.index, 'fmarriage'))

    # policy
    print('Getting column real_cancel')
    X_fs = X_fs.assign(real_cancel = get_bs2_real_cancel(df_policy, X_fs.index))

    print('Getting column cat_area')
    X_fs = X_fs.assign(cat_area = get_bs2_cat(df_policy, X_fs.index, 'iply_area'))

    print('Getting column cat_ic_combo')
    X_fs = X_fs.assign(cat_ic_combo = get_bs2_cat_ic_combo(df_policy, X_fs.index))

    print('Getting column cat_ic_grp_combo')
    X_fs = X_fs.assign(cat_ic_grp_combo = get_bs2_cat_ic_grp_combo(df_policy, X_fs.index))

    print('Getting column cat_distr')
    X_fs = X_fs.assign(cat_distr = get_bs2_cat(df_policy, X_fs.index, 'Distribution_Channel'))

    print('Getting column real_acc_dmg')
    X_fs = X_fs.assign(real_acc_dmg = get_bs2_cat(df_policy, X_fs.index, 'pdmg_acc'))

    print('Getting column real_acc_lia')
    X_fs = X_fs.assign(real_acc_lia = get_bs2_cat(df_policy, X_fs.index, 'plia_acc'))

    print('Getting column real_dage')
    X_fs = X_fs.assign(real_dage = get_bs2_real_dage(df_policy, X_fs.index))

    print('Getting column real_prem_terminate')
    X_fs = X_fs.assign(real_prem_terminate = get_bs2_real_prem_terminate(df_policy, X_fs.index))

    # vehicle
    print('Getting column cat_vmm1')
    X_fs = X_fs.assign(cat_vmm1 = get_bs2_cat(df_policy, X_fs.index, 'Vehicle_Make_and_Model1'))

    print('Getting column cat_vmm2')
    X_fs = X_fs.assign(cat_vmm2 = get_bs2_cat(df_policy, X_fs.index, 'Vehicle_Make_and_Model2'))

    print('Getting column real_vmy')
    X_fs = X_fs.assign(real_vmy = get_bs2_real_vmy(df_policy, X_fs.index))

    print('Getting column real_vengine')
    X_fs = X_fs.assign(real_vengine = get_bs2_cat(df_policy, X_fs.index, 'Engine_Displacement_(Cubic_Centimeter)'))

    print('Getting column cat_vregion')
    X_fs = X_fs.assign(cat_vregion = get_bs2_cat(df_policy, X_fs.index, 'Imported_or_Domestic_Car'))

    print('Getting column cat_vc')
    X_fs = X_fs.assign(cat_vc = get_bs2_cat(df_policy, X_fs.index, 'Coding_of_Vehicle_Branding_&_Type'))

    print('Getting column real_vqpt')
    X_fs = X_fs.assign(real_vqpt = get_bs2_cat(df_policy, X_fs.index, 'qpt'))

    print('Getting column real_vcost')
    X_fs = X_fs.assign(real_vcost = get_bs2_cat(df_policy, X_fs.index, 'Replacement_cost_of_insured_vehicle'))

    # claim
    print('Getting column real_num_claim')
    X_fs = X_fs.assign(real_num_claim = get_bs2_real_num_claim(df_claim, X_fs.index))

    print('Getting column real_nearest_claim')
    X_fs = X_fs.assign(real_nearest_claim = get_bs2_real_nearest_claim(df_claim, X_fs.index))

    print('Getting column cat_claim_cause')
    X_fs = X_fs.assign(cat_claim_cause = get_bs2_cat_claim_cause(df_claim, X_fs.index))

    print('Getting column real_loss')
    X_fs = X_fs.assign(real_loss = get_bs2_real_claim(df_claim, X_fs.index, 'Paid_Loss_Amount'))

    print('Getting column real_loss_ins')
    X_fs = X_fs.assign(real_loss_ins = get_bs2_real_loss_ins(df_policy, df_claim, X_fs.index))

    print('Getting column real_salvage')
    X_fs = X_fs.assign(real_salvage = get_bs2_real_claim(df_claim, X_fs.index, 'Salvage_or_Subrogation?'))

    print('Getting column real_claim_fault')
    X_fs = X_fs.assign(real_claim_fault = get_bs2_real_claim_fault(df_claim, X_fs.index))

    print('Getting column cat_claim_area')
    X_fs = X_fs.assign(cat_claim_area = get_bs2_cat_claim_area(df_claim, X_fs.index))

    print('Getting column real_claimants')
    X_fs = X_fs.assign(real_claimants = get_bs2_real_claimants(df_claim, X_fs.index))

    # helper columns
    print('Getting column real_prem_plc, for mean encoding use')
    X_fs = X_fs.assign(real_prem_plc = get_bs2_real_prem_plc(df_policy, X_fs.index))

    print('\nSplitting train valid test features\n')
    X_train_all = X_fs.loc[y_train_all.index]
    X_test = X_fs.loc[y_test.index]

    # add mean encoding on mean of diff btw next_premium and premium
    cols_cat = ['cat_vmm1', 'cat_vmm2', 'cat_vc']
    for col_cat in cols_cat:
        col_mean = col_cat.replace('cat_', 'real_mc_mean_diff_')
        print('Getting column ' + col_mean)
        X_test[col_mean] = get_bs2_real_mc_mean_diff(col_cat, X_train_all, y_train_all, X_valid=X_test, train_only=False, fold=5, prior=1000)
        X_train_all[col_mean] = get_bs2_real_mc_mean_diff(col_cat, X_train_all, y_train_all, X_valid=pd.DataFrame(), train_only=True, fold=5, prior=1000)

    # add mean encoding on mean of dividend btw next_premium and premium
    cols_cat = ['cat_claim_cause']
    for col_cat in cols_cat:
        col_mean = col_cat.replace('cat_', 'real_mc_mean_div_')
        print('Getting column ' + col_mean)
        X_test[col_mean] = get_bs2_real_mc_mean_div(col_cat, X_train_all, y_train_all, X_valid=X_test, train_only=False, fold=5, prior=1000)
        X_train_all[col_mean] = get_bs2_real_mc_mean_div(col_cat, X_train_all, y_train_all, X_valid=pd.DataFrame(), train_only=True, fold=5, prior=1000)

    # add median encoding on median of next_premium
    cols_cat = ['cat_ins', 'cat_assured', 'cat_sex', 'cat_distr', 'cat_ic_combo']
    for col_cat in cols_cat:
        col_median = col_cat.replace('cat_', 'real_mc_median_')
        print('Getting column ' + col_median)
        X_test[col_median] = get_bs2_real_mc_median(col_cat, X_train_all, y_train_all, X_valid=X_test, train_only=False, fold=5, prior=1000)
        X_train_all[col_median] = get_bs2_real_mc_median(col_cat, X_train_all, y_train_all, X_valid=pd.DataFrame(), train_only=True, fold=5, prior=1000)

    # add median encoding on median of diff btw next_premium and premium
    cols_cat = ['cat_assured', 'cat_sex']
    for col_cat in cols_cat:
        col_median = col_cat.replace('cat_', 'real_mc_median_diff_')
        print('Getting column ' + col_median)
        X_test[col_median] = get_bs2_real_mc_median_diff(col_cat, X_train_all, y_train_all, X_valid=X_test, train_only=False, fold=5, prior=1000)
        X_train_all[col_median] = get_bs2_real_mc_median_diff(col_cat, X_train_all, y_train_all, X_valid=pd.DataFrame(), train_only=True, fold=5, prior=1000)

    # add median encoding on median of div btw next_premium and premium
    cols_cat = ['cat_ins', 'cat_assured', 'cat_sex', 'cat_distr', 'cat_ic_combo', 'cat_ic_grp_combo', 'cat_area', 'cat_vregion']
    for col_cat in cols_cat:
        col_median = col_cat.replace('cat_', 'real_mc_median_div_')
        print('Getting column ' + col_median)
        X_test[col_median] = get_bs2_real_mc_median_div(col_cat, X_train_all, y_train_all, X_valid=X_test, train_only=False, fold=5, prior=1000)
        X_train_all[col_median] = get_bs2_real_mc_median_div(col_cat, X_train_all, y_train_all, X_valid=pd.DataFrame(), train_only=True, fold=5, prior=1000)

    # add mean encoding on probability of next_premium being 0
    cols_cat = ['cat_ins', 'cat_marriage', 'cat_claim_cause', 'cat_claim_area']
    for col_cat in cols_cat:
        col_prob = col_cat.replace('cat_', 'real_mc_prob_')
        print('Getting column ' + col_prob)
        X_test[col_prob] = get_bs2_real_mc_prob(col_cat, X_train_all, y_train_all, X_valid=X_test, train_only=False, fold=5, prior=1000)
        X_train_all[col_prob] = get_bs2_real_mc_prob(col_cat, X_train_all, y_train_all, X_valid=pd.DataFrame(), train_only=True, fold=5, prior=1000)

    print('Writing results to file')
    write_data(X_train_all, "X_train_all_bs2.csv")
    write_data(y_train_all, "y_train_all_bs2.csv")
    write_data(X_test, "X_test_bs2.csv")
    write_data(y_test, "y_test_bs2.csv")

    return(None)

######## get submission results ########
def get_bs2_quick_submission(params):
    '''
    In:

    Out:
        float(mae)

    Description:
        calculate quick mae on validation set
    '''
    X_train = read_data('X_train_all_bs2.csv')
    y_train = read_data('y_train_all_bs2.csv')
    X_valid = read_data('X_test_bs2.csv')
    y_valid = read_data('y_test_bs2.csv')

    cols_train = [col for col in X_train.columns if not col.startswith('cat')]
    All_train = y_train.merge(X_train, how='left', left_index=True, right_index=True)
    All_valid = y_valid.merge(X_valid, how='left', left_index=True, right_index=True)

    lgb_train = lgb.Dataset(All_train[cols_train].values, All_train['Next_Premium'].values.flatten(), free_raw_data=False)

    lgb_params = deepcopy(params)
    model = lgb.train(
        lgb_params['model'], lgb_train, **lgb_params['train']
    )

    valid_pred = model.predict(All_valid[cols_train])
    valid_pred = pd.DataFrame(valid_pred, index = All_valid.index)
    valid_pred.columns = ['Next_Premium']
    write_data(valid_pred, 'testing-set.csv')

    return(None)


def demo():
#    df_claim = read_data('claim_0702.csv', path='raw')
#    df_policy = read_data('policy_0702.csv', path='raw')
#    get_bs2_combined_features(df_policy, df_claim)

    lgb_model_params = {
        'boosting_type': 'gbdt',
        'num_iterations': 3000,
        'objective': 'regression_l1',
        'metric': 'mae',

#        'num_leaves': 31,
#        'max_depth': -1,
#        'min_data_in_leaf': 20,
#        'bagging_fraction': 1.0,
#        'bagging_freq': 0,
#        'feature_fraction': 1.0,
#        'max_bin': 255,
#        'min_data_in_bin': 3,
#        'max_delta_step': 0.0,
#        'lambda_l1': 0.0,
#        'lambda_l2': 0.0,
#        'min_gain_to_split': 0.0,
#        'bin_construct_sample_cnt': 220000,
        'num_leaves': 63,
        'min_data_in_leaf': 15,
        'min_gain_to_split': 0.01,
        'bin_construct_sample_cnt': 220000,

        'seed': 0,
        'bagging_seed': 3,
        'feature_fraction_seed': 2,
    }
    lgb_train_params = {
        'early_stopping_rounds': None,
        'learning_rates': lambda iter: max(0.01*(0.99**iter), 0.005),
        'verbose_eval': True,
    }
    lgb_params = {'model': lgb_model_params, 'train': lgb_train_params}
    get_bs2_quick_submission(lgb_params)

if __name__ == '__main__':
    fire.Fire(demo)