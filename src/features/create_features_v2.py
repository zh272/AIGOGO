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


def get_bs2_real_prem(df_policy, idx_df, col, mv=0):
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
    # remove earlier terminated value
    real_ia = df_policy['Insured_Amount1'] + df_policy['Insured_Amount2'] + df_policy['Insured_Amount3']
    df_policy = df_policy[real_ia != 0]
    # premium by policy
    df_policy = df_policy.groupby(level=0).agg({'Premium': np.sum, col: lambda x: x.iloc[0]})
    # premium by category
    df_map = df_policy.groupby([col]).agg({'Premium': np.mean})
    # map premium by category to policy
    real_prem_col = df_policy[col].map(df_map['Premium'])
    real_prem_col = real_prem_col.loc[idx_df]

    if real_prem_col.isnull().any().any():
        print('Premium on {} contains null values after removing early terminated'.format(col))
        print('{} will be inputed for missing value'.format(mv))

    return(real_prem_col.fillna(mv))

def get_bs2_real_prem_ic(df_policy, idx_df, col, mv=0):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(real_prem_ic),
    Description:
        get premium by insurance coverage
    '''
    # remove earlier terminated value
    real_ia = df_policy['Insured_Amount1'] + df_policy['Insured_Amount2'] + df_policy['Insured_Amount3']
    df_policy = df_policy[real_ia != 0]
    # get feature interaction
    cols_inter = ['Insurance_Coverage', col]
    # get map of premium to category interaction between ic and col
    df_map = df_policy.groupby(cols_inter).agg({'Premium': np.mean})
    # map category interaction mean premium to policy number
    real_prem_ic = df_policy[cols_inter].merge(df_map, how='left', left_on=cols_inter, right_index=True)
    real_prem_ic = real_prem_ic.groupby(level=0).agg({'Premium': np.sum})
    real_prem_ic = real_prem_ic.loc[idx_df, 'Premium']

    if real_prem_ic.isnull().any().any():
        print('Premium on {} contains null values after removing early terminated'.format(col))
        print('{} will be inputed for missing value'.format(mv))

    return(real_prem_ic.fillna(mv))


def get_bs2_real_mc_mean_diff(col_cat, X_train, y_train, X_valid=pd.DataFrame(), train_only=True, fold=5, prior=1000):
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
            X_slice = get_bs2_real_mc_mean_diff(col_cat, X_base, y_base, X_valid=X_slice, train_only=False, prior=prior)
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
        real_mc_mean_diff = real_mc_mean_diff.where(~pd.isnull(real_mc_mean_diff), np.mean(y_train['real_mc_mean_diff'])) + X_valid['real_prem_plc']

    return(real_mc_mean_diff)


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

    return(real_age.loc[idx_df])