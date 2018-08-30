import numpy as np
import os
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import NMF
from helpers import MultiColumnLabelEncoder
import lightgbm as lgb
import itertools

'''
Steps to add a new feature:

    (1) add your function definition to section ##feature explosion##; the function should take in (df_policy[, df_claim], index); Index instance: X_train.index

    (2) call your function in create_feature_selection_data(df_policy, df_claim)

    (3) ensure the main function uses lgm to select features, run the script, and move any features that are kicked out by feature selection to recycle bin
'''

######## recycle bin ########
'''
print('Getting column real_prem_vregion')
X_fs = X_fs.assign(real_prem_vregion = get_bs_real_prem_vregion(df_policy, X_fs.index))

'''

######## get feature selection ########
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


def type_feature_selection(get_fs_mae, params, num_only=False):
    '''
    In:
        any(get_fs_mae),
        dict(params),
        bool(num_only), # include numeric features only

    Out:
        dict(cols), # test mae and selected columns

    Description:
        feature selection by type
    '''
    X_train = read_interim_data('X_train_prefs.csv')
    y_train = read_interim_data('y_train_prefs.csv')
    X_valid = read_interim_data('X_valid_prefs.csv')
    y_valid = read_interim_data('y_valid_prefs.csv')
    # X_test = read_interim_data('X_test_prefs.csv')
    # y_test = read_interim_data('y_test_prefs.csv')

    # preprocessing
    X_train.fillna(-999, inplace=True)
    X_valid.fillna(-999, inplace=True)

    All_train = y_train.merge(X_train, how='left', left_index=True, right_index=True)
    All_valid = y_valid.merge(X_valid, how='left', left_index=True, right_index=True)

    def feature_selection_by_scenario(mae_min, cols_fs, lst_cols):
        print('Baseline data gives minimum mae {}'.format(mae_min))
        for i in range(len(lst_cols)):
            cols = lst_cols[i]

            print('Testing features {}'.format(cols))
            mae, cols_del = get_fs_mae(cols, All_train, All_valid, params)
            print('MAE changes to {}'.format(mae))

            cols_fs = cols if mae < mae_min else cols_fs
            mae_min = mae if mae < mae_min else mae_min

        return(mae_min, cols_fs)

    # baseline column
    cols_bs = ['real_prem_plc']
    #cols_bs = ['real_prem_plc', 'real_prem_dmg', 'real_prem_lia', 'real_prem_thf', 'real_prem_ic_nmf_1', 'real_prem_ic_nmf_2', 'real_prem_ic_nmf_3', 'real_prem_ic_nmf_4', 'real_prem_ic_nmf_5', 'real_prem_ic_nmf_6', 'real_prem_ic_nmf_7', 'real_freq_distr', 'real_prem_ic_distr', 'real_mc_mean_distr', 'real_mc_prob_distr', 'int_acc_lia', 'real_acc_dmg', 'real_acc_lia', 'real_mc_prob_cancel', 'real_mc_mean_age', 'real_mc_prob_age', 'real_mc_mean_marriage', 'real_mc_prob_marriage', 'real_mc_mean_vmy', 'real_mc_prob_vmy', 'real_vcost', 'real_mc_prob_area', 'real_mc_mean_claim_ins', 'real_mc_prob_claim_ins']
    #cols_bs = ['real_prem_plc', 'real_prem_dmg', 'real_prem_lia', 'real_prem_thf', 'real_prem_ic_nmf_1', 'real_prem_ic_nmf_2', 'real_prem_ic_nmf_3', 'real_prem_ic_nmf_4', 'real_prem_ic_nmf_5', 'real_prem_ic_nmf_6', 'real_prem_ic_nmf_7', 'real_mc_mean_distr', 'real_mc_prob_distr', 'real_acc_dmg', 'real_mc_mean_cancel', 'real_mc_prob_cancel', 'real_mc_mean_age', 'real_mc_prob_age', 'real_mc_mean_assured', 'real_mc_prob_assured', 'real_mc_mean_marriage', 'real_mc_prob_marriage', 'real_mc_mean_vmy', 'real_mc_prob_vmy', 'int_claim_plc', 'real_mc_mean_claim_ins', 'real_loss_plc', 'real_loss_ins']
    cols_fs = cols_bs
    mae_min, cols_del = get_fs_mae(cols_bs, All_train, All_valid, params)

    # premium selection
    # selected columns
    '''
    ['real_prem_plc',
     'real_prem_dmg',
     'real_prem_lia',
     'real_prem_thf',
     'real_prem_ic_nmf_1',
     'real_prem_ic_nmf_2',
     'real_prem_ic_nmf_3',
     'real_prem_ic_nmf_4',
     'real_prem_ic_nmf_5',
     'real_prem_ic_nmf_6',
     'real_prem_ic_nmf_7',
     ]
    '''
    print('\nSelecting features based on Premium by Insurance Coverage')

    cols_prem_grp = ['real_prem_dmg', 'real_prem_lia', 'real_prem_thf']
    cols_prem_nmf = ['real_prem_ic_nmf_' + str(i) for i in range(1, 8)]
    lst_cols_prem = [cols_prem_grp, cols_prem_nmf,
                     [col for lst in [cols_bs, cols_prem_grp] for col in lst],
                     [col for lst in [cols_bs, cols_prem_nmf] for col in lst],
                     [col for lst in [cols_bs, cols_prem_grp, cols_prem_nmf] for col in lst]]
    if num_only:
        lst_cols_prem = [[col for col in lst if not col.startswith('cat')] for lst in lst_cols_prem]

    mae_min, cols_fs = feature_selection_by_scenario(mae_min, cols_fs, lst_cols_prem)

    print('Premium by Insurance Coverage feature selection gives columns {}'.format(cols_fs))
    print('Premium by Insurance Coverage feature selection gives MAE {}'.format(mae_min))

    # distribution selection
    # selected columns
    '''
    ['real_freq_distr',
     'real_prem_ic_distr',
     'real_mc_mean_distr',
     'real_mc_prob_distr'
     ]
    '''
    print('\nSelecting features based on Distribution Channel')

    cols_distr = [col for col in X_train.columns if 'distr' in col]
    if num_only:
        cols_distr = [col for col in cols_distr if not col.startswith('cat')]

    lst_cols_distr = []
    for l in range(1, len(cols_distr) + 1):
        for lst in itertools.combinations(cols_distr, l):
            lst_cols_distr.append(cols_fs + list(lst))

    mae_min, cols_fs = feature_selection_by_scenario(mae_min, cols_fs, lst_cols_distr)

    print('Premium by Distribution Channel feature selection gives columns {}'.format(cols_fs))
    print('Premium by Distribution Channel feature selection gives MAE {}'.format(mae_min))

    # policy selection
    '''
    ['int_acc_lia',
    'real_acc_dmg',
    'real_acc_lia',
    'real_mc_prob_cancel'
    ]
    '''
    print('\nSelecting features based on Policy Information')
    cols_plc = ['int_acc_lia', 'real_acc_dmg', 'real_acc_lia', 'cat_cancel', 'real_mc_mean_cancel', 'real_mc_prob_cancel']
    if num_only:
        cols_plc = [col for col in cols_plc if not col.startswith('cat')]

    lst_cols_plc = []
    for l in range(1, len(cols_plc) + 1):
        for lst in itertools.combinations(cols_plc, l):
            lst_cols_plc.append(cols_fs + list(lst))

    mae_min, cols_fs = feature_selection_by_scenario(mae_min, cols_fs, lst_cols_plc)

    print('Premium by Policy feature selection gives columns {}'.format(cols_fs))
    print('Premium by Policy feature selection gives MAE {}'.format(mae_min))

    # insured selection
    '''
    ['real_mc_mean_age',
     'real_mc_prob_age',
     'real_mc_mean_marriage',
     'real_mc_prob_marriage',
     ]
    '''
    print('\nSelecting features based on Insured Information')

    cols_ins_age = ['cat_age', 'real_mc_mean_age', 'real_mc_prob_age']
    cols_ins_assured = ['cat_assured', 'real_mc_mean_assured', 'real_mc_prob_assured']
    cols_ins_marriage = ['cat_marriage', 'real_mc_mean_marriage', 'real_mc_prob_marriage']
    cols_ins_sex = ['cat_sex', 'real_mc_mean_sex', 'real_mc_prob_sex']
    cols_ins_sex_marr = ['cat_sex_marr', 'real_mc_mean_sex_marr', 'real_mc_prob_sex_marr']

    cols_ins = [cols_ins_age, cols_ins_assured, cols_ins_marriage, cols_ins_sex, cols_ins_sex_marr]

    if num_only:
        cols_ins = [[col for col in lst if not col.startswith('cat')] for lst in cols_ins]
    # pre-selection by category
    lst_cols_ins = []
    for l in range(1, len(cols_ins) + 1):
        for lst in itertools.combinations(cols_ins, l):
            comb_list = [col for sublist in lst for col in sublist]
            lst_cols_ins.append(cols_fs + list(comb_list))

    mae_inter, cols_inter = feature_selection_by_scenario(mae_min, cols_fs, lst_cols_ins)

    cols_ins = [col for col in cols_inter if col not in cols_fs]
    # selection by each feature
    lst_cols_ins = []
    for l in range(1, len(cols_ins) + 1):
        for lst in itertools.combinations(cols_ins, l):
            lst_cols_ins.append(cols_fs + list(lst))

    mae_min, cols_fs = feature_selection_by_scenario(mae_min, cols_fs, lst_cols_ins)

    print('Premium by Insured feature selection gives columns {}'.format(cols_fs))
    print('Premium by Insured feature selection gives MAE {}'.format(mae_min))

    # vehicle selection
    '''
    ['real_mc_mean_vmy',
     'real_mc_prob_vmy',
     'real_vcost'
     ]
    '''
    print('\nSelecting features based on Vehicle Information')

    cols_veh_vc = ['cat_vc', 'real_mc_mean_vc', 'real_mc_prob_vc']
    cols_veh_vmm1 = ['cat_vmm1', 'real_mc_mean_vmm1', 'real_mc_prob_vmm1']
    cols_veh_vmm2 = ['cat_vmm2', 'real_mc_mean_vmm2', 'real_mc_prob_vmm2']
    cols_veh_vmy = ['cat_vmy', 'real_mc_mean_vmy', 'real_mc_prob_vmy']
    cols_veh_vregion = ['cat_vregion', 'real_mc_mean_vregion', 'real_mc_prob_vregion']
    cols_veh_vqpt = ['cat_vqpt', 'real_mc_mean_vqpt', 'real_mc_prob_vqpt']
    cols_veh_vcost = ['real_vcost']
    cols_veh_ved = ['cat_ved', 'real_mc_mean_ved', 'real_mc_prob_ved', 'real_ved']
    cols_veh = [cols_veh_vc, cols_veh_vmm1, cols_veh_vmm2, cols_veh_vmy, cols_veh_vregion, cols_veh_vqpt, cols_veh_vcost, cols_veh_ved]

    if num_only:
        cols_veh = [[col for col in lst if not col.startswith('cat')] for lst in cols_veh]
    # pre-selection by category
    lst_cols_veh = []
    for l in range(1, len(cols_veh) + 1):
        for lst in itertools.combinations(cols_veh, l):
            comb_list = [col for sublist in lst for col in sublist]
            lst_cols_veh.append(cols_fs + list(comb_list))

    mae_inter, cols_inter = feature_selection_by_scenario(mae_min, cols_fs, lst_cols_veh)

    cols_veh = [col for col in cols_inter if col not in cols_fs]
    # selection by each feature
    lst_cols_veh = []
    for l in range(1, len(cols_veh) + 1):
        for lst in itertools.combinations(cols_veh, l):
            lst_cols_veh.append(cols_fs + list(lst))

    mae_min, cols_fs = feature_selection_by_scenario(mae_min, cols_fs, lst_cols_veh)

    print('Premium by Vehicle Information feature selection gives columns {}'.format(cols_fs))
    print('Premium by Vehicle Information feature selection gives MAE {}'.format(mae_min))

    # location selection
    '''
    ['real_mc_prob_area']
    '''
    print('\nSelecting features based on Location Information')

    cols_loc = ['cat_zip', 'cat_area', 'real_mc_mean_zip', 'real_mc_prob_zip', 'real_mc_mean_area', 'real_mc_prob_area']
    if num_only:
        cols_loc = [col for col in cols_loc if not col.startswith('cat')]

    lst_cols_loc = []
    for l in range(1, len(cols_loc) + 1):
        for lst in itertools.combinations(cols_loc, l):
            lst_cols_loc.append(cols_fs + list(lst))

    mae_min, cols_fs = feature_selection_by_scenario(mae_min, cols_fs, lst_cols_loc)

    print('Premium by Location Information feature selection gives columns {}'.format(cols_fs))
    print('Premium by Location Information feature selection gives MAE {}'.format(mae_min))

    # claim selection
    '''
    ['real_mc_mean_claim_ins',
     'real_mc_prob_claim_ins']
    '''
    print('\nSelecting features based on Claim Information')

    cols_clm = ['int_claim_plc', 'cat_claim_ins', 'real_mc_mean_claim_ins', 'real_mc_prob_claim_ins', 'real_loss_plc', 'real_loss_ins']
    if num_only:
        cols_clm = [col for col in cols_clm if not col.startswith('cat')]

    lst_cols_clm = []
    for l in range(1, len(cols_clm) + 1):
        for lst in itertools.combinations(cols_clm, l):
            lst_cols_clm.append(cols_fs + list(lst))

    mae_min, cols_fs = feature_selection_by_scenario(mae_min, cols_fs, lst_cols_clm)

    print('Premium by Claim Information feature selection gives columns {}'.format(cols_fs))
    print('Premium by Claim Information feature selection gives MAE {}'.format(mae_min))

    # garbage bin
    print('\nSelecting features based on Garbage Information')
    '''
    []->none
    '''
    cols_gb = ['int_others', 'cat_vdom', 'cat_vmotor', 'cat_vmm1_vmy', 'real_mc_mean_vdom', 'real_mc_prob_vdom', 'real_mc_mean_vmotor', 'real_mc_prob_vmotor', 'real_mc_mean_vmm1_vmy', 'real_mc_prob_vmm1_vmy', 'real_prem_ins', 'real_prem_vc', 'real_prem_ved', 'real_prem_vmm1', 'real_prem_vmm2', 'real_prem_vmy', 'real_prem_vqpt']
    if num_only:
        cols_gb = [col for col in cols_gb if not col.startswith('cat')]

    forward = True
    while(forward):
        forward = False
        for col_gb in cols_gb:
            print('Testing add column {}'.format(col_gb))
            cols_train = cols_fs + [col_gb]
            mae, cols_del = get_fs_mae(cols_train, All_train, All_valid, params)
            print('MAE changes to {}'.format(mae))
            # update col_change, cols_del, and mae_min if mae is lower
            forward = forward if mae >= mae_min else True
            cols_fs = cols_fs if mae >= mae_min else cols_fs + [col_gb]
            mae_min = mae_min if mae >= mae_min else mae

    print('Premium by Garbage Information feature selection gives columns {}'.format(cols_fs))
    print('Premium by Garbage Information feature selection gives MAE {}'.format(mae_min))

    return(cols_fs)


def stepwise_feature_selection(get_fs_mae, params, max_rounds=60, num_only=False, forward_step=False, backward_step=True, cols_init=[]):
    '''
    In:
        any(get_fs_mae),
        dict(params),
        int(max_rounds),
        bool(num_only), # include numeric features only
        bool(forward_step),
        bool(backward_step),
        list(cols_init),

    Out:
        dict(cols), # test mae and selected columns

    Description:
        do stepwise feature selection
    '''
    X_train = read_interim_data('X_train_prefs.csv')
    y_train = read_interim_data('y_train_prefs.csv')
    X_valid = read_interim_data('X_valid_prefs.csv')
    y_valid = read_interim_data('y_valid_prefs.csv')
    X_test = read_interim_data('X_test_prefs.csv')
    y_test = read_interim_data('y_test_prefs.csv')

    # preprocessing
    X_train.fillna(-999, inplace=True)
    X_valid.fillna(-999, inplace=True)

    All_train = y_train.merge(X_train, how='left', left_index=True, right_index=True)
    All_valid = y_valid.merge(X_valid, how='left', left_index=True, right_index=True)

    categorical_features = []
    for name in All_train.columns.values:
        if 'cat' in name or 'int' in name:
            categorical_features.append(name)
            All_train[name] = All_train[name].astype('category')
            All_valid[name] = All_valid[name].astype('category')


    le = MultiColumnLabelEncoder(columns=categorical_features)
    le.fit(pd.concat([All_train, All_valid]))
    All_train = le.transform(All_train)
    All_valid = le.transform(All_valid)

    # initialize add delete columns
    # h2o test 1: ['real_prem_plc', 'real_prem_ic_distr', 'real_prem_lia', 'cat_distr', 'int_acc_lia', 'cat_zip', 'cat_sex', 'real_acc_dmg', 'cat_vmy']
    # h2o test 2: real_prem_plc, real_prem_lia, cat_distr, int_acc_lia, cat_sex, real_acc_dmg, cat_zip
    if(len(cols_init) == 0):
        cols_init = list(X_train.columns)

    cols_train = cols_init
    cols_add = [col for col in X_train.columns if col not in cols_train]

    if num_only:
        cols_train = [col for col in cols_train if not col.startswith('cat')]
        cols_add = [col for col in cols_add if not col.startswith('cat')]

    # stepwise select features
    col_change = 'init'
    rounds = 0
    mae_min, cols_del = get_fs_mae(cols_train, All_train, All_valid, params)
    print('Baseline data gives minimum mae {}'.format(mae_min))

    while(col_change and rounds < max_rounds):
        # initialize
        col_change = None
        rounds = rounds + 1

        # backward step
        if backward_step and (len(cols_del) > 1):
            cols_del_new = cols_del
            for col_del in cols_del:
                print('Testing delete column {}'.format(col_del))
                cols_train_i = [col for col in cols_train if col != col_del]
                mae_i, cols_del_i = get_fs_mae(cols_train_i, All_train, All_valid, params)
                print('MAE changes to {}'.format(mae_i))
                # update col_change, cols_del, and mae_min if mae is lower
                col_change = col_change if mae_i >= mae_min else col_del
                cols_del_new = cols_del_new if mae_i >= mae_min else cols_del_i
                mae_min = mae_min if mae_i >= mae_min else mae_i
            cols_del = cols_del_new

        # forward step
        if forward_step and (len(cols_add) > 0):
            for col_add in cols_add:
                print('Testing add column {}'.format(col_add))
                cols_train_i = cols_train + [col_add]
                mae_i, cols_del_i = get_fs_mae(cols_train_i, All_train, All_valid, params)
                print('MAE changes to {}'.format(mae_i))
                # update col_change, cols_del, and mae_min if mae is lower
                col_change = col_change if mae_i >= mae_min else col_add
                cols_del = cols_del if mae_i >= mae_min else cols_del_i
                mae_min = mae_min if mae_i >= mae_min else mae_i

        if col_change in cols_add:
            print('Round {} adds column {}'.format(rounds, col_change))
            cols_train = cols_train + [col_change]
            cols_add = [col for col in cols_add if col != col_change]
        elif col_change in cols_train:
            print('Round {} removes column {}'.format(rounds, col_change))
            cols_add = cols_add + [col_change]
            cols_train = [col for col in cols_train if col != col_change]

        print('Existing {} columns are: {}'.format(len(cols_train), ', '.join(cols_train)))
        print('Round {} minimum mae is {}'.format(rounds, mae_min))

    X_train = pd.concat([X_train, X_valid])[cols_del]
    y_train = pd.concat([y_train, y_valid])
    X_test = X_test[cols_del]

    write_test_data(X_train, "X_train_fs.csv")
    write_test_data(y_train, "y_train_fs.csv")
    write_test_data(X_test, "X_test_fs.csv")
    write_test_data(y_test, "y_test_fs.csv")

    return(cols_train)


######## template function ########
def get_bs_cat(df_policy, idx_df, col):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df)
        str(col),
    Out:
        Series(cat_),
    Description:
        get category directly from df_policy
    '''
    df = df_policy.groupby(level=0).agg({col: lambda x: x.iloc[0]})
    return(df.loc[idx_df, col])


def get_bs_agg_premium(df_policy, idx_df, col):
    '''
    In:
        DataFrame(df_policy),
        str(col),
    Out:
        Series(real_prem_),
    Description:
        get premium aggregated on different level
    '''
    df = df_policy.groupby(level=0).agg({'Premium': np.nansum, col: lambda x: x.iloc[0]})
    df = df.groupby([col]).agg({'Premium': np.nanmedian})
    df = df_policy[[col]].merge(df, how='left', left_on=[col], right_index=True)
    df = df.groupby(level=0).agg({'Premium': lambda x: x.iloc[0]})
    df = df.assign(real_prem = df['Premium'].map(lambda x: 0 if pd.isnull(x) else x))
    return(df.loc[idx_df, 'real_prem'])


def get_bs_real_mc_mean(col_cat, X_train, y_train, X_valid=pd.DataFrame(), train_only=True, fold=5, prior=1000):
    '''
    In:
        str(col_cat)
        DataFrame(X_train),
        DataFrame(y_train),
        DataFrame(X_valid),
        bool(train_only),
        double(fold),
    Out:
        Series(real_mc_prob_distr),
    Description:
        get mean of next_premium by col_cat
    '''
    if train_only:
        np.random.seed(1)
        rand = np.random.rand(len(X_train))
        lvs = [i / float(fold) for i in range(fold+1)]

        X_arr = []
        for i in range(fold):
            msk = (rand >= lvs[i]) & (rand < lvs[i+1])
            X_slice = X_train[msk]
            X_base = X_train[~msk]
            y_base = y_train[~msk]
            X_slice = get_bs_real_mc_mean(col_cat, X_base, y_base, X_valid=X_slice, train_only=False, prior=prior)
            X_arr.append(X_slice)

        X_valid = pd.concat(X_arr)

    else:
        # merge col_cat with label
        y_train = y_train.merge(X_train[[col_cat]], how='left', left_index=True, right_index=True)
        y_train = y_train.assign(real_mc_mean = y_train['Next_Premium'])

        # get mean of each category and smoothed by global mean
        smooth_mean = lambda x: (x.sum() + prior * y_train['real_mc_mean'].mean()) / (len(x) + prior)
        y_train = y_train.groupby([col_cat]).agg({'real_mc_mean': smooth_mean})

        #
        X_valid = X_valid[[col_cat]].merge(y_train[['real_mc_mean']], how='left', left_on=[col_cat], right_index=True)
        X_valid = X_valid['real_mc_mean'].where(~pd.isnull(X_valid['real_mc_mean']), np.nanmean(y_train['real_mc_mean']))

    return(X_valid)


def get_bs_real_mc_prob(col_cat, X_train, y_train, X_valid=pd.DataFrame(), train_only=True, fold=5, prior=1000):
    '''
    In:
        str(col_cat)
        DataFrame(X_train),
        DataFrame(y_train),
        DataFrame(X_valid),
        bool(train_only),
        double(fold),
    Out:
        Series(real_mc_prob_distr),
    Description:
        get probability of premium reducing to 0 by col_cat
    '''
    if train_only:
        np.random.seed(1)
        rand = np.random.rand(len(X_train))
        lvs = [i / float(fold) for i in range(fold+1)]

        X_arr = []
        for i in range(fold):
            msk = (rand >= lvs[i]) & (rand < lvs[i+1])
            X_slice = X_train[msk]
            X_base = X_train[~msk]
            y_base = y_train[~msk]
            X_slice = get_bs_real_mc_prob(col_cat, X_base, y_base, X_valid=X_slice, train_only=False, prior=prior)
            X_arr.append(X_slice)

        X_valid = pd.concat(X_arr)

    else:
        # merge col_cat with label
        y_train = y_train.merge(X_train[[col_cat]], how='left', left_index=True, right_index=True)
        y_train = y_train.assign(real_mc_prob = y_train['Next_Premium'] != 0)

        # get mean of each category and smoothed by global mean
        smooth_mean = lambda x: (x.sum() + prior * y_train['real_mc_prob'].mean()) / (len(x) + prior)
        y_train = y_train.groupby([col_cat]).agg({'real_mc_prob': smooth_mean})

        X_valid = X_valid[[col_cat]].merge(y_train[['real_mc_prob']], how='left', left_on=[col_cat], right_index=True)
        X_valid = X_valid['real_mc_prob'].where(~pd.isnull(X_valid['real_mc_prob']), np.nanmean(y_train['real_mc_prob']))

    return(X_valid)


######## feature explosion ########
def get_bs_cat_age(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(cat_age),
    Description:
        get age label
    '''
    df        = df_policy.groupby(level=0).agg({'ibirth': lambda x: x.iloc[0]})
    get_label = lambda x: 0 if pd.isnull(x) else round((2015 - int(x[3:])) / 5)
    df        = df.assign(cat_age = df['ibirth'].map(get_label))
    return(df['cat_age'][idx_df])

def get_bs_cat_area(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(cat_area),
    Description:
        get area label
    '''
    return(get_bs_cat(df_policy, idx_df, 'iply_area'))


def get_bs_cat_assured(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(cat_assured),
    Description:
        get assured label
    '''
    return(get_bs_cat(df_policy, idx_df, 'fassured'))


def get_bs_cat_cancel(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(cat_cancel),
    Description:
        get cancel label
    '''
    df = df_policy.groupby(level=0).agg({'Cancellation': lambda x: x.iloc[0]})
    df = df.assign(cat_cancel = df['Cancellation'].map(lambda x: 1 if x=='Y' else 0))
    return(df['cat_cancel'][idx_df])


def get_bs_cat_distr(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(cat_distr),
    Description:
        get distribution channel label
    '''
    return(get_bs_cat(df_policy, idx_df, 'Distribution_Channel'))


def get_bs_cat_marriage(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(cat_marriage),
    Description:
        get marriage label
    '''
    df = df_policy.groupby(level=0).agg({'fmarriage': lambda x: x.iloc[0]})
    df = df.assign(cat_marriage = df['fmarriage'].map(lambda x: 0 if pd.isnull(x) else x))
    return(df['fmarriage'][idx_df])


def get_bs_cat_sex(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(cat_sex),
    Description:
        get sex label
    '''
    df = df_policy.groupby(level=0).agg({'fsex': lambda x: x.iloc[0]})
    df = df.assign(cat_sex = df['fsex'].map(lambda x: 0 if pd.isnull(x) else x))
    return(df['cat_sex'][idx_df])


def get_bs_cat_vc(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(cat_vc),
    Description:
        get vehicle code label
    '''
    return(get_bs_cat(df_policy, idx_df, 'Coding_of_Vehicle_Branding_&_Type'))


def get_bs_cat_vmm1(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(cat_vmm1),
    Description:
        get vehicle code label
    '''
    return(get_bs_cat(df_policy, idx_df, 'Vehicle_Make_and_Model1'))


def get_bs_cat_vmm2(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(cat_vmm2),
    Description:
        get vehicle code label
    '''
    return(get_bs_cat(df_policy, idx_df, 'Vehicle_Make_and_Model2'))


def get_bs_cat_vmy(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(cat_vmy),
    Description:
        get vehicle code label
    '''
    df = df_policy.groupby(level=0).agg({'Manafactured_Year_and_Month': lambda x: x.iloc[0]})
    get_label = lambda x: 2015 - x if x > 2010 else round((2015 - x) / 5 + 4)
    df = df.assign(cat_vmy = df['Manafactured_Year_and_Month'].map(get_label))
    return(df['cat_vmy'][idx_df])


def get_bs_cat_vqpt(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(cat_vqpt),
    Description:
        get vehicle code label
    '''
    df = df_policy.groupby(level=0).agg({'qpt': lambda x: x.iloc[0], 'fpt': lambda x: x.iloc[0]})
    df = df.assign(cat_vqpt = df['qpt'].map(lambda x: str(x)) + df['fpt'])
    return(df['cat_vqpt'][idx_df])


def get_bs_cat_vregion(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(cat_vregion),
    Description:
        get vehicle imported or domestic label
    '''
    return(get_bs_cat(df_policy, idx_df, 'Imported_or_Domestic_Car'))


def get_bs_cat_zip(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(cat_zip),
    Description:
        get assured zip label
    '''
    return(get_bs_cat(df_policy, idx_df, 'aassured_zip'))


def get_bs_int_acc_lia(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(int_acc_lia),
    Description:
        get liability class label
    '''
    return(get_bs_cat(df_policy, idx_df, 'lia_class'))


def get_bs_int_claim_plc(df_policy, df_claim, idx_df):
    '''
    In:
        DataFrame(df_policy),
        DataFrame(df_claim),
        Any(idx_df),
    Out:
        Series(int_claim_plc),
    Description:
        get number of claims on the policy
    '''
    df = df_claim.groupby(level=0).agg({'Claim_Number': lambda x: x.nunique()})
    df = df_policy.merge(df, how='left', left_index=True, right_index=True)
    df = df.groupby(level=0).agg({'Claim_Number': lambda x: x.iloc[0]})
    df = df.assign(int_claim_plc = df['Claim_Number'].map(lambda x: 0 if pd.isnull(x) else x))
    return(df['int_claim_plc'][idx_df])


def get_bs_int_others(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(int_others),
    Description:
        get number of other policies
    '''
    return(get_bs_cat(df_policy, idx_df, 'Multiple_Products_with_TmNewa_(Yes_or_No?)'))


def get_bs_real_acc_dmg(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(real_acc_dmg),
    Description:
        get liability class label
    '''
    return(get_bs_cat(df_policy, idx_df, 'pdmg_acc'))


def get_bs_real_acc_lia(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(real_acc_lia),
    Description:
        get liability class label
    '''
    return(get_bs_cat(df_policy, idx_df, 'plia_acc'))


def get_bs_real_loss_plc(df_policy, df_claim, idx_df):
    '''
    In:
        DataFrame(df_policy),
        DataFrame(df_claim),
        Any(idx_df),
    Out:
        Series(real_loss_plc),
    Description:
        get total loss of claims on the policy
    '''
    df = df_claim.groupby(level=0).agg({'Paid_Loss_Amount': np.nansum})
    df = df_policy.merge(df, how='left', left_index=True, right_index=True)
    df = df.groupby(level=0).agg({'Paid_Loss_Amount': lambda x: x.iloc[0]})
    df = df.assign(real_loss_plc = df['Paid_Loss_Amount'].map(lambda x: 0 if pd.isnull(x) else x))
    return(df['real_loss_plc'][idx_df])


def get_bs_real_prem_dmg(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(real_prem_dmg),
    Description:
        get premium on damage main category
    '''
    df = df_policy[df_policy['Main_Insurance_Coverage_Group']=='車損']
    df = df.groupby(level=0).agg({'Premium': np.nansum})
    df = df_policy[['Main_Insurance_Coverage_Group']].merge(df, how='left', left_index=True, right_index=True)
    df = df.groupby(level=0).agg({'Premium': lambda x: x.iloc[0]})
    df = df.assign(real_prem_dmg = df['Premium'].map(lambda x: 0 if pd.isnull(x) else x))
    return(df['real_prem_dmg'][idx_df])


def get_bs_real_prem_ins(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(real_prem_ins),
    Description:
        get premium by insured
    '''
    return(get_bs_agg_premium(df_policy, idx_df, "Insured's_ID"))


def get_bs_real_prem_lia(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(real_prem_lia),
    Description:
        get premium on liability main category
    '''
    df = df_policy[df_policy['Main_Insurance_Coverage_Group']=='車責']
    df = df.groupby(level=0).agg({'Premium': np.nansum})
    df = df_policy[['Main_Insurance_Coverage_Group']].merge(df, how='left', left_index=True, right_index=True)
    df = df.groupby(level=0).agg({'Premium': lambda x: x.iloc[0]})
    df = df.assign(real_prem_lia = df['Premium'].map(lambda x: 0 if pd.isnull(x) else x))
    return(df['real_prem_lia'][idx_df])


def get_bs_real_prem_plc(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(real_prem_plc),
    Description:
        get liability class label
    '''
    df = df_policy.groupby(level=0).agg({'Premium': np.nansum})
    return(df['Premium'][idx_df])


def get_bs_real_prem_thf(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(real_prem_thf),
    Description:
        get premium on liability main category
    '''
    df = df_policy[df_policy['Main_Insurance_Coverage_Group']=='竊盜']
    df = df.groupby(level=0).agg({'Premium': np.nansum})
    df = df_policy[['Main_Insurance_Coverage_Group']].merge(df, how='left', left_index=True, right_index=True)
    df = df.groupby(level=0).agg({'Premium': lambda x: x.iloc[0]})
    df = df.assign(real_prem_thf = df['Premium'].map(lambda x: 0 if pd.isnull(x) else x))
    return(df['real_prem_thf'][idx_df])


def get_bs_real_prem_vc(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(real_prem_vc),
    Description:
        get premium by insured
    '''
    return(get_bs_agg_premium(df_policy, idx_df, "Coding_of_Vehicle_Branding_&_Type"))


def get_bs_real_vcost(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(real_vcost),
    Description:
        get replacement cost
    '''
    return(get_bs_cat(df_policy, idx_df, 'Replacement_cost_of_insured_vehicle'))


def get_bs_real_ved(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(real_ved),
    Description:
        get engine displacement
    '''
    return(get_bs_cat(df_policy, idx_df, 'Engine_Displacement_(Cubic_Centimeter)'))


def get_bs_real_prem_vmm1(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(real_prem_vmm1),
    Description:
        get premium by vehicle model1
    '''
    return(get_bs_agg_premium(df_policy, idx_df, 'Vehicle_Make_and_Model1'))

def get_bs_real_prem_vmm2(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(real_prem_vmm2),
    Description:
        get premium by vehicle model2
    '''
    return(get_bs_agg_premium(df_policy, idx_df, 'Vehicle_Make_and_Model2'))


def get_bs_real_prem_vmy(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(real_prem_vmy),
    Description:
        get premium by vehicle manufacture year
    '''
    df = df_policy.assign(cat_vmy = get_bs_cat_vmy(df_policy, df_policy.index))
    return(get_bs_agg_premium(df, idx_df, 'cat_vmy'))


def get_bs_cat_ved(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(cat_ved),
    Description:
        cut engine displacement into 10 bins
    '''
    df = df_policy.groupby(level=0).agg({'Engine_Displacement_(Cubic_Centimeter)': lambda x: x.iloc[0]})
    df = pd.qcut(df['Engine_Displacement_(Cubic_Centimeter)'], 10, list(range(10)))
    return(df.loc[idx_df])


def get_bs_real_prem_ved(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(real_prem_ved),
    Description:
        get premium by engine displacement label
    '''
    df = df_policy.assign(cat_ved = get_bs_cat_ved(df_policy, df_policy.index))
    return(get_bs_agg_premium(df, idx_df, 'cat_ved'))


def get_bs_cat_vdom(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(cat_vdom),
    Description:
        get whether the vehicle is produced domestically
    '''
    df = df_policy.groupby(level=0).agg({'Imported_or_Domestic_Car': lambda x: x.iloc[0]})
    df = df['Imported_or_Domestic_Car'].map(lambda x: 1 if x == 10 else 0)
    return(df.loc[idx_df])


def get_bs_real_prem_vregion(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(real_prem_vregion),
    Description:
        get premium by manufacture location
    '''
    return(get_bs_agg_premium(df_policy, idx_df, 'Imported_or_Domestic_Car'))


def get_bs_cat_vmotor(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(cat_motor),
    Description:
        get vehicle type by fpt and qpt
    '''
    df = df_policy.groupby(level=0).agg({'qpt': lambda x: x.iloc[0], 'fpt': lambda x: x.iloc[0]})
    cat_vmotor = np.where((df['qpt'] <= 3) & (df['fpt'] == 'P'), 2, 1)
    cat_vmotor = np.where(df['fpt'] != 'P', 0, cat_vmotor)
    cat_vmotor = pd.Series(cat_vmotor, index=df.index)
    return(cat_vmotor.loc[idx_df])


def get_bs_real_prem_vqpt(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(real_prem_vqpt),
    Description:
        get premium by number of vehicle seats
    '''
    df = df_policy.assign(cat_vqpt = get_bs_cat_vqpt(df_policy, df_policy.index))
    return(get_bs_agg_premium(df, idx_df, 'cat_vqpt'))


def get_bs_cat_sex_marr(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(cat_sex_marr),
    Description:
        get cross category of sex and marriage
    '''
    df = df_policy.groupby(level=0).agg({'fsex': lambda x: x.iloc[0]})
    df = df.assign(cat_sex = get_bs_cat_sex(df_policy, df.index))
    df = df.assign(cat_marriage = get_bs_cat_marriage(df_policy, df.index))
    df = df['cat_sex'] + df['cat_marriage'] * 3
    return(df.loc[idx_df])


def get_bs_real_prem_distr(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(real_prem_distr),
    Description:
        get premium by distribution channel
    '''
    df = df_policy.groupby(["Distribution_Channel"]).agg({'Premium': np.nanmedian})
    df = df_policy[["Distribution_Channel"]].merge(df, how='left', left_on=["Distribution_Channel"], right_index=True)
    df = df.groupby(level=0).agg({'Premium': lambda x: x.iloc[0]})
    df = df.assign(real_prem_distr = df['Premium'].map(lambda x: 0 if pd.isnull(x) else x))
    return(df['real_prem_distr'][idx_df])


    def get_bs_real_prem_ic_distr(df_policy, idx_df):
        '''
        In:
            DataFrame(df_policy),
            Any(idx_df),
        Out:
            Series(real_prem_ic_distr),
        Description:
            get premium by distribution channel * insurance coverage
        '''
        df = df_policy.groupby(['Insurance_Coverage', "Distribution_Channel"]).agg({'Premium': np.nanmedian})
        df = df_policy[['Insurance_Coverage', "Distribution_Channel"]].merge(df, how='left', left_on=['Insurance_Coverage', "Distribution_Channel"], right_index=True)
        df = df.groupby(level=0).agg({'Premium': np.nansum})
        df = df.assign(real_prem_ic_distr = df['Premium'].map(lambda x: 0 if pd.isnull(x) else x))
        return(df['real_prem_ic_distr'][idx_df])


def get_bs_real_prem_area_distr(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(real_prem_area_distr),
    Description:
        get premium by distribution channel * insurance coverage
    '''
    df = df_policy.groupby(['iply_area', "Distribution_Channel"]).agg({'Premium': np.nanmedian})
    df = df_policy[['iply_area', "Distribution_Channel"]].merge(df, how='left', left_on=['iply_area', "Distribution_Channel"], right_index=True)
    df = df.groupby(level=0).agg({'Premium': np.nansum})
    df = df.assign(real_prem_area_distr = df['Premium'].map(lambda x: 0 if pd.isnull(x) else x))
    return(df['real_prem_area_distr'][idx_df])


def get_bs_real_freq_distr(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(real_freq_distr),
    Description:
        get frequency of distribution channel occurance
    '''
    df = df_policy.groupby(level=0).agg({'Distribution_Channel': lambda x: x.iloc[0]})
    df = pd.DataFrame(df['Distribution_Channel'].value_counts(normalize=True))
    df.columns = ['real_freq_distr']
    df = df_policy.merge(df, how='left', left_on=['Distribution_Channel'], right_index=True)
    df = df.groupby(level=0).agg({'real_freq_distr': lambda x: x.iloc[0]})
    return(df['real_freq_distr'][idx_df])


def get_bs_cat_vmm1_vmy(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(cat_vmm1_vmy),
    Description:
        get frequency of distribution channel occurance
    '''
    df = df_policy.groupby(level=0).agg({'Vehicle_Make_and_Model1': lambda x: x.iloc[0], 'Manafactured_Year_and_Month': lambda x: str(x.iloc[0])})
    df = df.assign(cat_vmm1_vmy = df['Vehicle_Make_and_Model1'] +  df['Manafactured_Year_and_Month'])

    return(df.loc[idx_df, 'cat_vmm1_vmy'])


def get_bs_cat_claim_ins(df_policy, df_claim, idx_df):
    '''
    In:
        DataFrame(df_policy),
        DataFrame(df_claim),
        Any(idx_df),
    Out:
        Series(cat_claim_ins),
    Description:
        get whether the insured got into an accident by himself
    '''
    df_claim = df_claim[df_claim["Driver's_Relationship_with_Insured"]==1]
    df_claim = df_claim.groupby(level=0).agg({"Driver's_Relationship_with_Insured": lambda x: 1})
    df = df_policy.merge(df_claim, how='left', left_index=True, right_index=True)
    df = df.groupby(level=0).agg({"Driver's_Relationship_with_Insured": lambda x: x.iloc[0]})
    df = df.loc[idx_df, "Driver's_Relationship_with_Insured"].fillna(0)

    return(df)


def get_bs_real_loss_ins(df_policy, df_claim, idx_df):
    '''
    In:
        DataFrame(df_policy),
        DataFrame(df_claim),
        Any(idx_df),
    Out:
        Series(real_loss_ins),
    Description:
        get total loss of claims given an insured
    '''
    df_claim = df_claim.groupby(level=0).agg({'Paid_Loss_Amount': np.nansum})
    df_policy = df_policy.groupby(level=0).agg({"Insured's_ID": lambda x: x.iloc[0]})
    df = df_policy.merge(df_claim, how='left', left_index=True, right_index=True)
    df_ins = df.groupby(["Insured's_ID"]).agg({'Paid_Loss_Amount': np.nansum})
    df = df[["Insured's_ID"]].merge(df_ins, how='left', left_on=["Insured's_ID"], right_index=True)
    df = df.loc[idx_df, 'Paid_Loss_Amount'].fillna(0)

    return(df)


def get_bs_real_prem_ic_nmf(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        DataFrame(real_prem_ic_nmf),
    Description:
        get premium by insurance coverage, with nonnegative matrix factorization
    '''
    df = df_policy[['Insurance_Coverage', 'Premium']]
    df = df.set_index('Insurance_Coverage', append=True)
    df = df.unstack(level=1)
    mtx_df = df.fillna(0).as_matrix()
    nmf_df = NMF(n_components=7, random_state=1, alpha=.1, l1_ratio=.5).fit_transform(mtx_df)
    df = pd.DataFrame(nmf_df, index = df.index)
    df.columns = ['real_prem_ic_nmf_' + str(i) for i in range(1, 8)]

    return(df.loc[idx_df])


######## get pre feature selection data set ########
def create_feature_selection_data(df_policy, df_claim):
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
    #X_bs = read_interim_data('X_train_bs.csv')
    #y_bs = read_interim_data('y_train_bs.csv')

    X_train = read_interim_data('X_train_bs.csv')
    X_test = read_interim_data('X_test_bs.csv')

    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    y_train = read_raw_data('training-set.csv')
    y_test = read_raw_data('testing-set.csv')

    X_fs = pd.concat([X_train, X_test])
    y_fs = pd.concat([y_train, y_test])

    # basic
    print('Getting column cat_sex_marr')
    X_fs = X_fs.assign(cat_sex_marr = get_bs_cat_sex_marr(df_policy, X_fs.index))

    # distribution
    print('Getting column real_freq_distr')
    X_fs = X_fs.assign(real_freq_distr = get_bs_real_freq_distr(df_policy, X_fs.index))

    print('Getting column real_prem_area_distr')
    X_fs = X_fs.assign(real_prem_area_distr = get_bs_real_prem_area_distr(df_policy, X_fs.index))

    print('Getting column real_prem_ic_distr')
    X_fs = X_fs.assign(real_prem_ic_distr = get_bs_real_prem_ic_distr(df_policy, X_fs.index))

    print('Getting column real_prem_distr')
    X_fs = X_fs.assign(real_prem_distr = get_bs_real_prem_distr(df_policy, X_fs.index))

    # vehicle
    print('Getting column cat_vdom')
    X_fs = X_fs.assign(cat_vdom = get_bs_cat_vdom(df_policy, X_fs.index))

    print('Getting column cat_ved')
    X_fs = X_fs.assign(cat_ved = get_bs_cat_ved(df_policy, X_fs.index))

    print('Getting column cat_vmotor')
    X_fs = X_fs.assign(cat_vmotor = get_bs_cat_vmotor(df_policy, X_fs.index))

    print('Getting column cat_vmm1_vmy')
    X_fs = X_fs.assign(cat_vmm1_vmy = get_bs_cat_vmm1_vmy(df_policy, X_fs.index))

    print('Getting column real_prem_ved')
    X_fs = X_fs.assign(real_prem_ved = get_bs_real_prem_ved(df_policy, X_fs.index))

    print('Getting column real_prem_vmm1')
    X_fs = X_fs.assign(real_prem_vmm1 = get_bs_real_prem_vmm1(df_policy, X_fs.index))

    print('Getting column real_prem_vmm2')
    X_fs = X_fs.assign(real_prem_vmm2 = get_bs_real_prem_vmm2(df_policy, X_fs.index))

    print('Getting column real_prem_vmy')
    X_fs = X_fs.assign(real_prem_vmy = get_bs_real_prem_vmy(df_policy, X_fs.index))

    print('Getting column real_prem_vqpt')
    X_fs = X_fs.assign(real_prem_vqpt = get_bs_real_prem_vqpt(df_policy, X_fs.index))

    # claim
    print('Getting column cat_claim_ins')
    X_fs = X_fs.assign(cat_claim_ins = get_bs_cat_claim_ins(df_policy, df_claim, X_fs.index))

    print('Getting column real_loss_ins')
    X_fs = X_fs.assign(real_loss_ins = get_bs_real_loss_ins(df_policy, df_claim, X_fs.index))

    # insurance coverage
    print('Getting column real_prem_ic_nmf')
    real_prem_ic_nmf = get_bs_real_prem_ic_nmf(df_policy, X_fs.index)
    colnames = ['real_prem_ic_nmf_' + str(i) for i in range(1, 8)]
    X_fs[colnames] = real_prem_ic_nmf

    # write results
    X_train = X_fs.loc[X_train.index]
    y_train = y_fs.loc[y_train.index]
    X_test = X_fs.loc[X_test.index]
    y_test = y_fs.loc[y_test.index]

    np.random.seed(0)
    msk = np.random.rand(len(X_train)) < 0.8
    X_train_v = X_train[~msk]
    y_train_v = y_train[~msk]
    X_train_t = X_train[msk]
    y_train_t = y_train[msk]

    # add mean encoding
    cols_cat = [col for col in X_train.columns if col.startswith('cat')]

    # add mean encoding on next_premium mean
    for col_cat in cols_cat:
        col_mean = col_cat.replace('cat_', 'real_mc_mean_')
        print('Getting column ' + col_mean)
        X_test[col_mean] = get_bs_real_mc_mean(col_cat, X_train, y_train, X_valid=X_test, train_only=False, fold=5, prior=1000)
        X_train_v[col_mean] = get_bs_real_mc_mean(col_cat, X_train_t, y_train_t, X_valid=X_train_v, train_only=False, fold=5, prior=1000)
        X_train_t[col_mean] = get_bs_real_mc_mean(col_cat, X_train, y_train, X_valid=pd.DataFrame(), train_only=True, fold=5, prior=1000)

    # add mean encoding on probability of next_premium being 0
    for col_cat in cols_cat:
        col_prob = col_cat.replace('cat_', 'real_mc_prob_')
        print('Getting column ' + col_prob)
        X_test[col_prob] = get_bs_real_mc_prob(col_cat, X_train, y_train, X_valid=X_test, train_only=False, fold=5, prior=1000)
        X_train_v[col_prob] = get_bs_real_mc_prob(col_cat, X_train_t, y_train_t, X_valid=X_train_v, train_only=False, fold=5, prior=1000)
        X_train_t[col_prob] = get_bs_real_mc_prob(col_cat, X_train, y_train, X_valid=pd.DataFrame(), train_only=True, fold=5, prior=1000)

    write_test_data(X_train_t, "X_train_prefs.csv")
    write_test_data(y_train_t, "y_train_prefs.csv")
    write_test_data(X_train_v, "X_valid_prefs.csv")
    write_test_data(y_train_v, "y_valid_prefs.csv")
    write_test_data(X_test, "X_test_prefs.csv")
    write_test_data(y_test, "y_test_prefs.csv")

    return(None)


######## read/write func ########
def read_raw_data(file_name, index_col='Policy_Number'):
    '''
    In: file_name

    Out: raw_data

    Description: read data from directory /data/raw
    '''
    # set the path of raw data
    raw_data_path = os.path.join(os.getcwd(), os.path.pardir, os.path.pardir, 'data', 'raw')

    file_path = os.path.join(raw_data_path, file_name)
    raw_data = pd.read_csv(file_path, index_col=index_col)

    return(raw_data)


def read_interim_data(file_name, index_col='Policy_Number'):
    '''
    In: file_name

    Out: interim_data

    Description: read data from directory /data/interim
    '''
    # set the path of raw data
    interim_data_path = os.path.join(os.getcwd(), os.path.pardir, os.path.pardir, 'data', 'interim')

    file_path = os.path.join(interim_data_path, file_name)
    interim_data = pd.read_csv(file_path, index_col=index_col)

    return(interim_data)


def write_test_data(df, file_name):
    '''
    In:
        DataFrame(df),
        str(file_name),
    Out:
        None

    Description:
        Write sample data to directory /data/interim
    '''
    interim_data_path = os.path.join(os.getcwd(), os.path.pardir, os.path.pardir, 'data', 'interim')
    write_sample_path = os.path.join(interim_data_path, file_name)
    df.to_csv(write_sample_path)

    return(None)


if __name__ == '__main__':
    '''
    train data: training-set.csv
    test data: testing-set.csv
    independent_claim: claim_0702.csv
    independent_policy: policy_0702.csv
    '''

    df_train = read_raw_data('training-set.csv')
    df_test = read_raw_data('testing-set.csv')
    df_claim = read_raw_data('claim_0702.csv')
    df_policy = read_raw_data('policy_0702.csv')

    #create_feature_selection_data(df_policy, df_claim)

    cat_params = {
        'n_estimators':100000, 'learning_rate': 50, 'objective':'MAE', 'verbose':False,
        'max_depth':4, 'colsample_bylevel':0.7, 'reg_lambda':None, 'task_type': 'CPU'
    }

    lgb_model_params = {
        'boosting_type': 'gbdt',
        'num_iterations': 1000,
        'max_depth':-1,
        'objective': 'regression_l1',
        'metric': 'mae',
        'lamba_l1':0.3,
        'num_leaves': 31,
        'learning_rate': 0.05,
        'colsample_bytree': 0.9,
        'subsample': 0.8,
        'subsample_freq': 5,
        'min_data_in_leaf': 20,
        'min_gain_to_split': 0,
        'seed': 0,
    }
    lgb_train_params = {
        'early_stopping_rounds': 3,
        'learning_rates': None, # lambda iter: 0.1*(0.99**iter),
        'verbose_eval': False,
    }
    lgb_params = {'model': lgb_model_params, 'train': lgb_train_params}

    #stepwise_feature_selection(get_lgb_mae, lgb_params, max_rounds=60, num_only=True, forward_step=False, backward_step=True, cols_init=[])

    type_feature_selection(get_lgb_mae, lgb_params, num_only=True)