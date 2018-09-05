import numpy as np
import pandas as pd

######## feature template expansion ########
def get_bs2_cat(df_policy, idx_df, col, mv=0):
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
    df = df.loc[idx_df, col]

    if df.isnull().any().any():
        print('{} contains null values'.format(col))
        print('{} will be inputed for missing value'.format(mv))

    return(df.fillna(mv))


def get_bs2_real_mc_mean(col_cat, X_train, y_train, X_valid=pd.DataFrame(), train_only=True, fold=5, prior=1000):
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
            X_slice = get_bs2_real_mc_mean(col_cat, X_base, y_base, X_valid=X_slice, train_only=False, prior=prior)
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


def get_bs2_real_mc_prob(col_cat, X_train, y_train, X_valid=pd.DataFrame(), train_only=True, fold=5, prior=1000):
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
            X_slice = get_bs2_real_mc_prob(col_cat, X_base, y_base, X_valid=X_slice, train_only=False, prior=prior)
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


def get_bs2_real_age(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(real_age),
    Description:
        get inssured age
    '''
    df_policy = df_policy.groupby(level=0).agg({'ibirth': lambda x: x.iloc[0]})

    get_real_age = lambda x: 0 if pd.isnull(x) else 2016 - int(x[3:])
    real_age = df_policy['ibirth'].map(get_real_age)

    return(real_age)


def get_bs2_real_age_grp(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(real_age),
    Description:
        get inssured age
    '''
    df_policy = df_policy.groupby(level=0).agg({'ibirth': lambda x: x.iloc[0]})

    get_real_age = lambda x: 0 if pd.isnull(x) else round((2016 - int(x[3:])) / 5)
    real_age = df_policy['ibirth'].map(get_real_age)

    return(real_age)


def get_bs2_real_age_tail(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(real_age),
    Description:
        get inssured age
    '''
    df_policy = df_policy.groupby(level=0).agg({'ibirth': lambda x: x.iloc[0]})

    get_real_age = lambda x: 0 if pd.isnull(x) else 2016 - int(x[3:])
    real_age = df_policy['ibirth'].map(get_real_age)

    get_age_tail = lambda x: 80 if x > 75 else x
    real_age = real_age.map(get_age_tail)

    get_age_head = lambda x: 20 if x < 25 and x > 0 else x
    real_age = real_age.map(get_age_head)

    return(real_age)


def get_bs2_real_prem_plc(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(real_prem_plc),
    Description:
    '''
    real_prem_plc = df_policy.groupby(level=0).agg({'Premium': np.nansum})
    return(real_prem_plc.loc[idx_df])