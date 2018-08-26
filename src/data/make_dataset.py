import numpy as np
import os
import pandas as pd

######## feature template ########
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

    return(df.loc[idx_df, col].fillna(0))


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
        y_train = y_train.assign(real_mc_mean_diff = y_train['Next_Premium'] / y_train['real_prem_plc'])

        # get mean of each category and smoothed by global mean
        smooth_mean = lambda x: (x.sum() + prior * y_train['real_mc_mean_diff'].mean()) / (len(x) + prior)
        y_train = y_train.groupby([col_cat]).agg({'real_mc_mean_diff': smooth_mean})
        real_mc_mean_diff = X_valid[col_cat].map(y_train['real_mc_mean_diff'])
        # fill na with global mean
        real_mc_mean_diff = real_mc_mean_diff.where(~pd.isnull(real_mc_mean_diff), np.mean(y_train['real_mc_mean_diff'])) * X_valid['real_prem_plc']

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


######## manual feature ########
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
    df_policy = df_policy.groupby(level=0).agg({'Manafactured_Year_and_Month': lambda x: x.iloc[0]})
    get_label = lambda x: 2015 - x if x > 2010 else round((2016 - x) / 5 + 4)
    cat_vmy = df_policy['Manafactured_Year_and_Month'].map(get_label)
    return(cat_vmy.loc[idx_df].fillna(0))


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
    real_prem_plc = df_policy.groupby(level=0).agg({'Premium': np.nansum})
    return(real_prem_plc.loc[idx_df])


######## get dataset aggregate ########
def create_train_test_data_bs(df_train, df_test, df_policy, df_claim):
    '''
    In:
        DataFrame(df_train), # contains dependent variable
        DataFrame(df_test),
        DataFrame(df_policy),
        DataFrame(df_claim),

    Out:
        DataFrame(df_sample),

    Description:
        create X, and y variables for training and testing
    '''
    # baseline columns

    '''
    X_train = read_interim_data('X_train_bs.csv')
    X_test = read_interim_data('X_test_bs.csv')
    df_bs = pd.concat([X_train, X_test])
    '''
    df_bs = pd.concat([df_train, df_test])

    print('Getting column cat_cancel')
    df_bs = df_bs.assign(cat_cancel = get_bs_cat(df_policy, df_bs.index, 'Cancellation'))

    print('Getting column cat_distr')
    df_bs = df_bs.assign(cat_distr = get_bs_cat(df_policy, df_bs.index, 'Distribution_Channel'))

    print('Getting column cat_ins')
    df_bs = df_bs.assign(cat_ins = get_bs_cat(df_policy, df_bs.index, "Insured's_ID"))

    print('Getting column cat_marriage')
    df_bs = df_bs.assign(cat_marriage = get_bs_cat(df_policy, df_bs.index, 'fmarriage'))

    print('Getting column cat_sex')
    df_bs = df_bs.assign(cat_sex = get_bs_cat(df_policy, df_bs.index, 'fsex'))

    print('Getting column cat_vmy')
    df_bs = df_bs.assign(cat_vmy = get_bs_cat_vmy(df_policy, df_bs.index))

    print('Getting column cat_acc_lia')
    df_bs = df_bs.assign(cat_acc_lia = get_bs_cat(df_policy, df_bs.index, 'lia_class'))

    print('Getting column real_acc_dmg')
    df_bs = df_bs.assign(real_acc_dmg = get_bs_cat(df_policy, df_bs.index, 'pdmg_acc'))

    print('Getting column real_acc_lia')
    df_bs = df_bs.assign(real_acc_lia = get_bs_cat(df_policy, df_bs.index, 'plia_acc'))

    print('Getting column real_prem_plc')
    df_bs = df_bs.assign(real_prem_plc = get_bs_real_prem_plc(df_policy, df_bs.index))

    print('Getting column real_vcost')
    df_bs = df_bs.assign(real_vcost = get_bs_cat(df_policy, df_bs.index, 'Replacement_cost_of_insured_vehicle'))

    # feature template expansion
    cols_cat = [col for col in df_bs.columns if col.startswith('cat')]

    # frequency of category values
    for col_cat in cols_cat:
        col_freq = col_cat.replace('cat_', 'real_freq_')
        print('Getting column ' + col_freq)
        df_bs[col_freq] = get_bs_real_freq(df_bs, df_bs.index, col_cat)

    col_y = 'Next_Premium'
    cols_X = [col for col in df_bs.columns if col != col_y]

    np.random.seed(0)
    msk = np.random.rand(len(df_train)) < 0.8
    df_valid = df_train[~msk]
    df_train = df_train[msk]

    X_train = df_bs.loc[df_train.index, cols_X]
    X_valid = df_bs.loc[df_valid.index, cols_X]
    X_test = df_bs.loc[df_test.index, cols_X]

    # add mean encoding
    # add mean encoding on next_premium mean
    for col_cat in cols_cat:
        col_mean = col_cat.replace('cat_', 'real_mc_mean_')
        print('Getting column ' + col_mean)
        X_test[col_mean] = get_bs_real_mc_mean(col_cat, X_train, df_train, X_valid=X_test, train_only=False, fold=5, prior=1000)
        X_valid[col_mean] = get_bs_real_mc_mean(col_cat, X_train, df_train, X_valid=X_valid, train_only=False, fold=5, prior=1000)
        X_train[col_mean] = get_bs_real_mc_mean(col_cat, X_train, df_train, X_valid=pd.DataFrame(), train_only=True, fold=5, prior=1000)

#    # add mean encoding on mean of diff btw next_premium and premium
#    for col_cat in cols_cat:
#        col_mean = col_cat.replace('cat_', 'real_mc_mean_diff_')
#        print('Getting column ' + col_mean)
#        X_test[col_mean] = get_bs_real_mc_mean_diff(col_cat, X_train, df_train, X_valid=X_test, train_only=False, fold=5, prior=1000)
#        X_valid[col_mean] = get_bs_real_mc_mean_diff(col_cat, X_train, df_train, X_valid=X_valid, train_only=False, fold=5, prior=1000)
#        X_train[col_mean] = get_bs_real_mc_mean_diff(col_cat, X_train, df_train, X_valid=pd.DataFrame(), train_only=True, fold=5, prior=1000)

    # add mean encoding on probability of next_premium being 0
    for col_cat in cols_cat:
        col_prob = col_cat.replace('cat_', 'real_mc_prob_')
        print('Getting column ' + col_prob)
        X_test[col_prob] = get_bs_real_mc_prob(col_cat, X_train, df_train, X_valid=X_test, train_only=False, fold=5, prior=1000)
        X_valid[col_prob] = get_bs_real_mc_prob(col_cat, X_train, df_train, X_valid=X_valid, train_only=False, fold=5, prior=1000)
        X_train[col_prob] = get_bs_real_mc_prob(col_cat, X_train, df_train, X_valid=pd.DataFrame(), train_only=True, fold=5, prior=1000)

    write_test_data(X_train, "X_train_bs.csv")
    write_test_data(X_valid, "X_valid_bs.csv")
    write_test_data(X_test, "X_test_bs.csv")
    write_test_data(df_train, "y_train_bs.csv")
    write_test_data(df_valid, "y_valid_bs.csv")
    write_test_data(df_test, "y_test_bs.csv")

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

    df_train = read_raw_data('training-set.csv')
    df_test = read_raw_data('testing-set.csv')
    df_claim = read_raw_data('claim_0702.csv')
    df_policy = read_raw_data('policy_0702.csv')

    create_train_test_data_bs(df_train, df_test, df_policy, df_claim)

