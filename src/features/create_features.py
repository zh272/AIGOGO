import numpy as np
import pandas as pd
import os
from sklearn.decomposition import NMF
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

######## feature template expansion ########
def get_bs_cat(df_policy, idx_df, col):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
        str(col),
    Out:
        Series(cat_),
    Description:
        get category directly from df_policy
    '''
    df = df_policy.groupby(level=0).agg({col: lambda x: x.iloc[0]})

    return(df.loc[idx_df, col])


def get_bs_real_prem(df_policy, idx_df, col):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df)
        str(col),
    Out:
        Series(real_prem_),
    Description:
        get feature interaction between categorical features and premium
    '''
    # premium by policy
    df_policy = df_policy.groupby(level=0).agg({'Premium': np.sum, col: lambda x: x.iloc[0]})
    # premium by category
    df_map = df_policy.groupby([col]).agg({'Premium': np.mean})
    # map premium by category to policy
    real_prem_col = df_policy[col].map(df_map['Premium'])

    return(real_prem_col.loc[idx_df])


def get_bs_real_prem_ic(df_policy, idx_df, col):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(real_prem_ic),
    Description:
        get premium by insurance coverage
    '''
    cols_inter = ['Insurance_Coverage', col]
    # get map of premium to category interaction between ic and col
    df_map = df_policy.groupby(cols_inter).agg({'Premium': np.mean})
    # map category interaction mean premium to policy number
    real_prem_ic = df_policy[cols_inter].merge(df_map, how='left', left_on=cols_inter, right_index=True)
    real_prem_ic = real_prem_ic.groupby(level=0).agg({'Premium': np.sum})

    return(real_prem_ic.loc[idx_df, 'Premium'])


def get_bs_real_freq(X_all, idx_df, col):
    '''
    In:
        DataFrame(X_all),
        Any(idx_df)
        str(col),
    Out:
        Series(real_freq_),
    Description:
        get number of occurance of each value of categorical features
    '''
    # frequency of category
    df_map = X_all.groupby([col]).agg({'real_prem_plc': lambda x: len(x)})
    # map premium by category to policy
    real_freq_col = X_all[col].map(df_map['real_prem_plc'])

    return(real_freq_col.loc[idx_df])


def get_bs_cat_inter(df_policy, idx_df, col1, col2):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df)
        str(col),
    Out:
        Series(cat_col1_col2),
    Description:
        get interaction of two categorical features
    '''
    # all col combination of col1 and col2
    df_policy = df_policy.groupby(level=0).agg({col1: lambda x: str(x.iloc[0]), col2: lambda x: str(x.iloc[0])})
    # concat col1 and col2
    cat_col1_col2 = df_policy[col1] + df_policy[col2]

    return(cat_col1_col2.loc[idx_df])


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
        real_mc_mean = pd.concat(X_arr).loc[X_train.index]

    else:
        # merge col_cat with label
        y_train = y_train.merge(X_train[[col_cat]], how='left', left_index=True, right_index=True)
        y_train = y_train.assign(real_mc_mean = y_train['Next_Premium'])

        # get mean of each category and smoothed by global mean
        smooth_mean = lambda x: (x.sum() + prior * y_train['real_mc_mean'].mean()) / (len(x) + prior)
        y_train = y_train.groupby([col_cat]).agg({'real_mc_mean': smooth_mean})
        real_mc_mean = X_valid[col_cat].map(y_train['real_mc_mean'])
        # fill na with global mean
        real_mc_mean = real_mc_mean.where(~pd.isnull(real_mc_mean), np.mean(y_train['real_mc_mean']))

    return(real_mc_mean)


def get_bs_real_mc_mean_diff(col_cat, X_train, y_train, X_valid=pd.DataFrame(), train_only=True, fold=5, prior=1000):
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
            X_slice = get_bs_real_mc_mean_diff(col_cat, X_base, y_base, X_valid=X_slice, train_only=False, prior=prior)
            X_arr.append(X_slice)
        real_mc_mean_diff = pd.concat(X_arr).loc[X_train.index]

    else:
        # merge col_cat with label
        y_train = y_train.merge(X_train[[col_cat, 'real_prem_plc']], how='left', left_index=True, right_index=True)
        y_train = y_train.assign(real_mc_mean_diff = y_train['Next_Premium'] - y_train['real_prem_plc'])

        # get mean of each category and smoothed by global mean
        smooth_mean = lambda x: (x.sum() + prior * y_train['real_mc_mean_diff'].mean()) / (len(x) + prior)
        y_train = y_train.groupby([col_cat]).agg({'real_mc_mean_diff': smooth_mean})
        real_mc_mean_diff = X_valid[col_cat].map(y_train['real_mc_mean_diff'])
        # fill na with global mean
        real_mc_mean_diff = real_mc_mean_diff.where(~pd.isnull(real_mc_mean_diff), np.mean(y_train['real_mc_mean_diff']))

    return(real_mc_mean_diff)


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
        Series(real_mc_prob),
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
        real_mc_prob = pd.concat(X_arr).loc[X_train.index]

    else:
        # merge col_cat with label
        y_train = y_train.merge(X_train[[col_cat]], how='left', left_index=True, right_index=True)
        y_train = y_train.assign(real_mc_prob = y_train['Next_Premium'] != 0)

        # get mean of each category and smoothed by global mean
        smooth_mean = lambda x: (x.sum() + prior * y_train['real_mc_prob'].mean()) / (len(x) + prior)
        y_train = y_train.groupby([col_cat]).agg({'real_mc_prob': smooth_mean})
        real_mc_prob = X_valid[col_cat].map(y_train['real_mc_prob'])
        # fill na with global mean
        real_mc_prob = real_mc_prob.where(~pd.isnull(real_mc_prob), np.mean(y_train['real_mc_prob']))

    return(real_mc_prob)


######## feature manual expansion ########
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
    # get whether the insured person is involved in the accident
    cat_claim_ins = df_claim.groupby(level=0).agg({"Driver's_Relationship_with_Insured": lambda x: x.iloc[0] == 1})
    cat_claim_ins = cat_claim_ins.loc[idx_df].fillna(False)

    return(cat_claim_ins)


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
    # get insured's id by policy
    df_policy = df_policy.groupby(level=0).agg({"Insured's_ID": lambda x: x.iloc[0]})
    # get paid loss by policy
    df_claim = df_claim.groupby(level=0).agg({'Paid_Loss_Amount': np.sum})
    # map claim paid loss to all policy
    df_map = df_policy.merge(df_claim, how='left', left_index=True, right_index=True)
    df_map = df_map.groupby(["Insured's_ID"]).agg({'Paid_Loss_Amount': np.sum})
    # get paid loss by insured's id
    real_loss_ins = df_policy["Insured's_ID"].map(df_map['Paid_Loss_Amount'])
    real_loss_ins = real_loss_ins.loc[idx_df].fillna(0)

    return(real_loss_ins)


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
    # rows: policy number; cols: insurance coverage
    df_policy = df_policy.set_index('Insurance_Coverage', append=True)
    df_policy = df_policy[['Premium']].unstack(level=1)
    # transform dataframe to matrix
    mtx_df = df_policy.fillna(0).as_matrix()
    # non-negative matrix factorization
    nmf_df = NMF(n_components=7, random_state=1, alpha=.1, l1_ratio=.5).fit_transform(mtx_df)
    #
    real_prem_ic_nmf = pd.DataFrame(nmf_df, index = df_policy.index)
    real_prem_ic_nmf.columns = ['real_prem_ic_nmf_' + str(i) for i in range(1, 8)]

    return(real_prem_ic_nmf.loc[idx_df])


def get_bs_real_prem_terminate(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        DataFrame(real_prem_terminate),
    Description:
        get premium on early terminated contracts
    '''
    # filter early termination
    real_ia_total = df_policy['Insured_Amount1'] + df_policy['Insured_Amount2'] + df_policy['Insured_Amount3']
    df_policy = df_policy[real_ia_total == 0]
    # sum up premium with early termination
    df_policy = df_policy.groupby(level=0).agg({'Premium': np.sum})
    # fillna and select index
    real_prem_terminate = df_policy.loc[idx_df, 'Premium'].fillna(0)

    return(real_prem_terminate)


def get_bs_cat_ins_self(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df)
        str(col),
    Out:
        Series(cat_ins_self),
    Description:
        get whether insured's birth equals to buyer's birth
    '''
    df_policy = df_policy.groupby(level=0).agg({'ibirth': lambda x: x.iloc[0], 'dbirth': lambda x: x.iloc[0]})
    cat_ins_self = df_policy['ibirth'] == df_policy['ibirth']

    return(cat_ins_self.loc[idx_df])


def get_bs_cat_claim_theft(df_claim, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df)
        str(col),
    Out:
        Series(cat_claim_theft),
    Description:
        get whether insured's birth equals to buyer's birth
    '''
    ic_theft = ['05N', '09@', '09I', '10A', '68E', '68N']
    df_claim = df_claim.assign(cat_claim_theft = df_claim['Coverage'].map(lambda x: 1 if x in ic_theft else 0))
    df_claim = df_claim.groupby(level=0).agg({'cat_claim_theft': np.max})
    cat_claim_theft = df_claim.loc[idx_df, 'cat_claim_theft'].fillna(0)

    return(cat_claim_theft)


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
    print('Getting column cat_ins_self')
    X_fs = X_fs.assign(cat_ins_self = get_bs_cat_ins_self(df_policy, X_fs.index))

    # distribution
    print('Getting column real_prem_ic_distr')
    X_fs = X_fs.assign(real_prem_ic_distr = get_bs_real_prem_ic(df_policy, X_fs.index, 'Distribution_Channel'))

    # vehicle
    print('Getting column real_prem_ic_vmy')
    X_fs = X_fs.assign(real_prem_ic_vmy = get_bs_real_prem_ic(df_policy, X_fs.index, 'Main_Insurance_Coverage_Group'))

    print('Getting column real_prem_per_vcost')
    X_fs = X_fs.assign(real_prem_per_vcost = X_fs['real_prem_plc'] / X_fs['real_vcost'])

    # claim
    print('Getting column cat_claim_ins')
    X_fs = X_fs.assign(cat_claim_ins = get_bs_cat_claim_ins(df_policy, df_claim, X_fs.index))

    print('Getting column real_loss_ins')
    X_fs = X_fs.assign(real_loss_ins = get_bs_real_loss_ins(df_policy, df_claim, X_fs.index))

    print('Getting column cat_claim_theft')
    X_fs = X_fs.assign(cat_claim_theft = get_bs_cat_claim_theft(df_claim, X_fs.index))

    # insurance coverage
    print('Getting column real_prem_ic_nmf')
    colnames = ['real_prem_ic_nmf_' + str(i) for i in range(1, 8)]
    X_fs[colnames] = get_bs_real_prem_ic_nmf(df_policy, X_fs.index)

    print('Getting column real_prem_terminate')
    X_fs = X_fs.assign(real_prem_terminate = get_bs_real_prem_terminate(df_policy, X_fs.index))

    # feature template expansion
    cols_cat = [col for col in X_fs.columns if col.startswith('cat')]

    # frequency of category values
    for col_cat in cols_cat:
        col_freq = col_cat.replace('cat_', 'real_freq_')
        print('Getting column ' + col_freq)
        X_fs[col_freq] = get_bs_real_freq(X_fs, X_fs.index, col_cat)

    # train valid test split
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
    # add mean encoding on next_premium mean
    for col_cat in cols_cat:
        col_mean = col_cat.replace('cat_', 'real_mc_mean_')
        print('Getting column ' + col_mean)
        X_test[col_mean] = get_bs_real_mc_mean(col_cat, X_train, y_train, X_valid=X_test, train_only=False, fold=5, prior=1000)
        X_train_v[col_mean] = get_bs_real_mc_mean(col_cat, X_train_t, y_train_t, X_valid=X_train_v, train_only=False, fold=5, prior=1000)
        X_train_t[col_mean] = get_bs_real_mc_mean(col_cat, X_train, y_train, X_valid=pd.DataFrame(), train_only=True, fold=5, prior=1000)

#    # add mean encoding on mean of diff btw next_premium and premium
#    for col_cat in cols_cat:
#        col_mean = col_cat.replace('cat_', 'real_mc_mean_diff_')
#        print('Getting column ' + col_mean)
#        X_test[col_mean] = get_bs_real_mc_mean_diff(col_cat, X_train, y_train, X_valid=X_test, train_only=False, fold=5, prior=1000)
#        X_train_v[col_mean] = get_bs_real_mc_mean_diff(col_cat, X_train_t, y_train_t, X_valid=X_train_v, train_only=False, fold=5, prior=1000)
#        X_train_t[col_mean] = get_bs_real_mc_mean_diff(col_cat, X_train, y_train, X_valid=pd.DataFrame(), train_only=True, fold=5, prior=1000)

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


def get_bs_quick_mae(params):
    '''
    In:

    Out:
        float(mae)

    Description:
        calculate quick mae on validation set
    '''
    X_train = read_interim_data('X_train_prefs.csv')
    y_train = read_interim_data('y_train_prefs.csv')
    X_valid = read_interim_data('X_valid_prefs.csv')
    y_valid = read_interim_data('y_valid_prefs.csv')

    # preprocessing
    X_train.fillna(-999, inplace=True)
    X_valid.fillna(-999, inplace=True)

    cols_train = [col for col in X_train.columns if not col.startswith('cat')]

    All_train = y_train.merge(X_train, how='left', left_index=True, right_index=True)
    All_valid = y_valid.merge(X_valid, how='left', left_index=True, right_index=True)

    lgb_train = lgb.Dataset(All_train[cols_train].values, All_train['Next_Premium'].values.flatten(), free_raw_data=False)
    lgb_valid = lgb.Dataset(All_valid[cols_train].values, All_valid['Next_Premium'].values.flatten(), reference=lgb_train, free_raw_data=False)

    model = lgb.train(
        params['model'], lgb_train, valid_sets=lgb_valid, **params['train']
    )

    valid_pred = model.predict(All_valid[cols_train])
    valid_mae = mean_absolute_error(All_valid['Next_Premium'], valid_pred)

    print('pre-selection mae is {}'.format(valid_mae))

    return(None)


######## read/write func ########
def read_raw_data(file_name, index_col='Policy_Number'):
    '''
    In: file_name

    Out: raw_data

    Description: read data from directory /data/raw
    '''
    # set the path of raw data
    raw_data_path = os.path.join(os.path.dirname('__file__'), os.path.pardir, os.path.pardir, 'data', 'raw')

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
    interim_data_path = os.path.join(os.path.dirname('__file__'), os.path.pardir, os.path.pardir, 'data', 'interim')

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
    interim_data_path = os.path.join(os.path.dirname('__file__'), os.path.pardir, os.path.pardir, 'data', 'interim')
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

    df_claim = read_raw_data('claim_0702.csv')
    df_policy = read_raw_data('policy_0702.csv')

    create_feature_selection_data(df_policy, df_claim)

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
    get_bs_quick_mae(lgb_params)