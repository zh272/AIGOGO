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


def get_bs2_real_prem_ic(df_policy, idx_df, df_key):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(real_prem_ic),
    Description:
        get premium by insurance coverage
    '''
    # filter early termination
    real_ia_total = df_policy['Insured_Amount1'] + df_policy['Insured_Amount2'] + df_policy['Insured_Amount3']
    df_policy = df_policy[real_ia_total != 0]
    # merge key to policy
    df_policy = df_policy.loc[idx_df].merge(pd.DataFrame(df_key, columns=['key']), how='left', left_index=True, right_index=True)
    cols_inter = ['Insurance_Coverage', 'key']
    # get map of premium to category interaction between ic and col
    df_map = df_policy.groupby(cols_inter).agg({'Premium': np.mean})
    # map category interaction mean premium to policy number
    real_prem_ic = df_policy[cols_inter].merge(df_map, how='left', left_on=cols_inter, right_index=True)
    real_prem_ic = real_prem_ic.groupby(level=0).agg({'Premium': np.sum})

    return(real_prem_ic.loc[idx_df, 'Premium'].fillna(0))


def get_bs2_real_prem_terminate(df_policy, idx_df):
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


def get_bs2_real_mc_mean_div(col_cat, X_train, y_train, X_valid=pd.DataFrame(), train_only=True, fold=5, prior=1000):
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
            X_slice = get_bs2_real_mc_mean_div(col_cat, X_base, y_base, X_valid=X_slice, train_only=False, prior=prior)
            X_arr.append(X_slice)
        real_mc_mean_div = pd.concat(X_arr).loc[X_train.index]

    else:
        # merge col_cat with label
        y_train = y_train.merge(X_train[[col_cat, 'real_prem_plc']], how='left', left_index=True, right_index=True)
        y_train = y_train.assign(real_mc_mean_div = y_train['Next_Premium'] / y_train['real_prem_plc'])

        # get mean of each category and smoothed by global mean
        smooth_mean = lambda x: (x.sum() + prior * y_train['real_mc_mean_div'].mean()) / (len(x) + prior)
        y_train = y_train.groupby([col_cat]).agg({'real_mc_mean_div': smooth_mean})
        real_mc_mean_div = X_valid[col_cat].map(y_train['real_mc_mean_div'])
        # fill na with global mean
        real_mc_mean_div = real_mc_mean_div.where(~pd.isnull(real_mc_mean_div), np.mean(y_train['real_mc_mean_div'])) * X_valid['real_prem_plc']

    return(real_mc_mean_div)


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


def get_bs2_real_mc_median(col_cat, X_train, y_train, X_valid=pd.DataFrame(), train_only=True, fold=5, prior=1000):
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
        get median of next_premium by col_cat
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
            X_slice = get_bs2_real_mc_median(col_cat, X_base, y_base, X_valid=X_slice, train_only=False, prior=prior)
            X_arr.append(X_slice)
        real_mc_median = pd.concat(X_arr).loc[X_train.index]

    else:
        # merge col_cat with label
        y_train = y_train.merge(X_train[[col_cat]], how='left', left_index=True, right_index=True)
        y_train = y_train.assign(real_mc_median = y_train['Next_Premium'])

        # get median of each category and smoothed by global median
        smooth_median = lambda x: (x.sum() + prior * y_train['real_mc_median'].median()) / (len(x) + prior)
        y_train = y_train.groupby([col_cat]).agg({'real_mc_median': smooth_median})
        real_mc_median = X_valid[col_cat].map(y_train['real_mc_median'])
        # fill na with global median
        real_mc_median = real_mc_median.where(~pd.isnull(real_mc_median), np.median(y_train['real_mc_median']))

    return(real_mc_median)


def get_bs2_real_mc_median_div(col_cat, X_train, y_train, X_valid=pd.DataFrame(), train_only=True, fold=5, prior=1000):
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
        get median of next_premium by col_cat
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
            X_slice = get_bs2_real_mc_median_div(col_cat, X_base, y_base, X_valid=X_slice, train_only=False, prior=prior)
            X_arr.append(X_slice)
        real_mc_median_div = pd.concat(X_arr).loc[X_train.index]

    else:
        # merge col_cat with label
        y_train = y_train.merge(X_train[[col_cat, 'real_prem_plc']], how='left', left_index=True, right_index=True)
        y_train = y_train.assign(real_mc_median_div = y_train['Next_Premium'] / y_train['real_prem_plc'])

        # get median of each category and smoothed by global median
        smooth_median = lambda x: (x.sum() + prior * y_train['real_mc_median_div'].median()) / (len(x) + prior)
        y_train = y_train.groupby([col_cat]).agg({'real_mc_median_div': smooth_median})
        real_mc_median_div = X_valid[col_cat].map(y_train['real_mc_median_div'])
        # fill na with global median
        real_mc_median_div = real_mc_median_div.where(~pd.isnull(real_mc_median_div), np.median(y_train['real_mc_median_div'])) * X_valid['real_prem_plc']

    return(real_mc_median_div)


def get_bs2_real_mc_median_diff(col_cat, X_train, y_train, X_valid=pd.DataFrame(), train_only=True, fold=5, prior=1000):
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
        get median of next_premium by col_cat
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
            X_slice = get_bs2_real_mc_median_diff(col_cat, X_base, y_base, X_valid=X_slice, train_only=False, prior=prior)
            X_arr.append(X_slice)
        real_mc_median_diff = pd.concat(X_arr).loc[X_train.index]

    else:
        # merge col_cat with label
        y_train = y_train.merge(X_train[[col_cat, 'real_prem_plc']], how='left', left_index=True, right_index=True)
        y_train = y_train.assign(real_mc_median_diff = y_train['Next_Premium'] - y_train['real_prem_plc'])

        # get median of each category and smoothed by global median
        smooth_median = lambda x: (x.sum() + prior * y_train['real_mc_median_diff'].median()) / (len(x) + prior)
        y_train = y_train.groupby([col_cat]).agg({'real_mc_median_diff': smooth_median})
        real_mc_median_diff = X_valid[col_cat].map(y_train['real_mc_median_diff'])
        # fill na with global median
        real_mc_median_diff = real_mc_median_diff.where(~pd.isnull(real_mc_median_diff), np.median(y_train['real_mc_median_diff'])) + X_valid['real_prem_plc']

    return(real_mc_median_diff)


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


def get_bs2_real_cancel(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(cat_),
    Description:
        get category directly from df_policy
    '''
    df = df_policy.groupby(level=0).agg({'Cancellation': lambda x: x.iloc[0]})
    df = df.loc[idx_df, 'Cancellation']
    df = np.where(df == 'Y', 1, 0)

    return(df)


def get_bs2_cat_ic_combo(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(cat_ic_combo),
    Description:
        get insured coverage combination
    '''
    df_policy = df_policy.sort_values(by=['Premium'], ascending=False)
    df_policy = df_policy.groupby(level=0).agg({'Insurance_Coverage': lambda x: '|'.join(np.sort(x[:3]))})
    cat_ic_combo = df_policy.loc[idx_df, 'Insurance_Coverage']

    return(cat_ic_combo)


def get_bs2_cat_ic_grp_combo(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(cat_ic_grp_combo),
    Description:
        get insured coverage combination
    '''
    df_policy = df_policy.groupby(level=0).agg({'Main_Insurance_Coverage_Group': lambda x: '|'.join(np.sort(x.unique()))})
    cat_ic_grp_combo = df_policy.loc[idx_df, 'Main_Insurance_Coverage_Group']

    return(cat_ic_grp_combo)


def get_bs2_real_dage(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(real_dage),
    Description:
        get inssured age
    '''
    df_policy = df_policy.groupby(level=0).agg({'dbirth': lambda x: x.iloc[0]})

    get_real_age = lambda x: 0 if pd.isnull(x) else 2016 - int(x[3:])
    real_age = df_policy['dbirth'].map(get_real_age)

    return(real_age)


def get_bs2_real_vmy(df_policy, idx_df):
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
    get_label = lambda x: -1 if pd.isnull(x) else 2016 - x
    real_vmy = df_policy['Manafactured_Year_and_Month'].map(get_label)

    return(real_vmy.loc[idx_df].fillna(-1))


def get_bs2_real_vmy_tail(df_policy, idx_df):
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
    get_label = lambda x: -1 if pd.isnull(x) else 2016 - x
    real_vmy = df_policy['Manafactured_Year_and_Month'].map(get_label)

    get_vmy_tail = lambda x: 25 if x > 20 else x
    real_vmy = real_vmy.map(get_vmy_tail)

    return(real_vmy.loc[idx_df].fillna(-1))


def get_bs2_real_vengine_grp(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(real_vengine_grp),
    Description:
        get vehicle code label
    '''
    df_policy = df_policy.groupby(level=0).agg({'Engine_Displacement_(Cubic_Centimeter)': lambda x: x.iloc[0]})
    grp_bins = [-1, 0, 500, 600, 1200, 1800, 2400, 3000, 3600, 4200, 4800, 5400, 6200, 6800, 7400, 8000, 1000000]
    real_vengine_grp = df_policy.loc[idx_df, 'Engine_Displacement_(Cubic_Centimeter)'].fillna(-1)
    real_vengine_grp = pd.cut(real_vengine_grp, bins = grp_bins, labels = False)

    return(real_vengine_grp)


def get_bs2_real_vequip(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df)
        str(col),
    Out:
        Series(real_vequip),
    Description:
        get whether the vehicle has additional equipment
    '''
    # get whether the vehicle has additional equipment
    cols_vequip = ['fequipment' + str(i) for i in list(range(1, 10)) if i not in [7, 8]]
    real_vequip = df_policy[cols_vequip].sum(axis=1)
    real_vequip.columns = ['real_vequip']
    # group by policy number
    real_vequip = real_vequip.groupby(level=0).agg({'real_vequip': lambda x: x.iloc[0]})
    return(real_vequip.loc[idx_df])


def get_bs2_real_num_claim(df_claim, idx_df):
    '''
    In:
        DataFrame(df_policy),
        DataFrame(df_claim),
        Any(idx_df),
    Out:
        Series(real_num_claim),
    Description:
        get total loss of claims given an insured
    '''
    df_claim = df_claim.groupby(level=0).agg({'Claim_Number': lambda x: len(x.unique())})
    real_num_claim = df_claim.loc[idx_df, 'Claim_Number'].fillna(0)

    return(real_num_claim)


def get_bs2_real_claim_self(df_policy, df_claim, idx_df):
    '''
    In:
        DataFrame(df_policy),
        DataFrame(df_claim),
        Any(idx_df),
    Out:
        Series(real_claim_ins),
    Description:
        get whether the insured got into an accident by himself
    '''
    # merge df_policy to df_claim
    df_policy = df_policy.groupby(level=0).agg({'fsex': lambda x: x.iloc[0], 'ibirth': lambda x: x.iloc[0], 'fmarriage': lambda x: x.iloc[0]})
    df_claim = df_claim.groupby(level=0).agg({"Driver's_Gender": lambda x: x.iloc[0], 'DOB_of_Driver': lambda x: x.iloc[0], "Driver's_Relationship_with_Insured": lambda x: x.iloc[0], 'Marital_Status_of_Driver': lambda x: x.iloc[0]})
    df_claim = df_claim.merge(df_policy, how='left', left_index=True, right_index=True)
    # get whether the insured person is involved in the accident
    real_claim_ins = (df_claim["Driver's_Relationship_with_Insured"] == 1) & (df_claim['ibirth'] == df_claim['DOB_of_Driver']) & (df_claim['fmarriage'] == df_claim['Marital_Status_of_Driver']) & (df_claim['fsex'] == df_claim["Driver's_Gender"])
    real_claim_ins.columns = ['real_claim_ins']
    real_claim_ins = real_claim_ins.groupby(level=0).agg({'real_claim_ins': lambda x: 1 if x.any() else 0})
    real_claim_ins = real_claim_ins.loc[idx_df].fillna(-1)

    return(real_claim_ins)


def get_bs2_real_nearest_claim(df_claim, idx_df):
    '''
    In:
        DataFrame(df_policy),
        DataFrame(df_claim),
        Any(idx_df),
    Out:
        Series(real_num_claim),
    Description:
        get total loss of claims given an insured
    '''
    acci_date = df_claim['Accident_Date'].map(lambda x: (int(x[:4]) - 2015) * 12 + int(x[5:]))
    acci_date = acci_date.groupby(level=0).agg({'Accident_Date': lambda x: x.max()})
    acci_date = acci_date.loc[idx_df, 'Accident_Date'].fillna(-1)

    return(acci_date)


def get_bs2_cat_claim_cause(df_claim, idx_df):
    '''
    In:
        DataFrame(df_claim),
        Any(idx_df),
    Out:
        Series(real_loss),
    Description:
        get total loss of claims given an insured
    '''
    # get paid loss by policy
    df_claim = df_claim.groupby(level=0).agg({'Cause_of_Loss': lambda x: '|'.join([y[:5] for y in np.sort(x.unique())])})
    # get paid loss by insured's id
    cat_claim_cause = df_claim.loc[idx_df, 'Cause_of_Loss'].fillna('none')

    return(cat_claim_cause)


def get_bs2_real_claim(df_claim, idx_df, col):
    '''
    In:
        DataFrame(df_claim),
        Any(idx_df),
    Out:
        Series(real_loss),
    Description:
        get total loss of claims given an insured
    '''
    # get paid loss by policy
    df_claim = df_claim.groupby(level=0).agg({col: np.sum})
    # get paid loss by insured's id
    real_loss = df_claim.loc[idx_df, col].fillna(-1)

    return(real_loss)


def get_bs2_real_loss_ins(df_policy, df_claim, idx_df):
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
    real_loss_ins = real_loss_ins.loc[idx_df].fillna(-1)

    return(real_loss_ins)


def get_bs2_real_claim_fault(df_claim, idx_df):
    '''
    In:
        DataFrame(df_claim),
        Any(idx_df)
        str(col),
    Out:
        Series(real_claim_fault),
    Description:
        get the average fault in accidents
    '''
    df_claim = df_claim.groupby(['Policy_Number', 'Claim_Number']).agg({'At_Fault?': lambda x: x.iloc[0]})
    df_claim = df_claim.groupby(level=0).agg({'At_Fault?': np.nanmax})
    real_claim_fault = df_claim.loc[idx_df, 'At_Fault?'].fillna(-1)

    return(real_claim_fault)


def get_bs2_cat_claim_area(df_claim, idx_df):
    '''
    In:
        DataFrame(df_claim),
        Any(idx_df)
        str(col),
    Out:
        Series(real_claim_area),
    Description:
        get the average fault in accidents
    '''
    # get paid loss by policy
    df_claim = df_claim.groupby(level=0).agg({'Accident_area': lambda x: '|'.join([y[:5] for y in np.sort(x.unique())])})
    # get paid loss by insured's id
    cat_claim_area = df_claim.loc[idx_df, 'Accident_area'].fillna('none')

    return(cat_claim_area)


def get_bs2_real_claimants(df_claim, idx_df):
    '''
    In:
        DataFrame(df_claim),
        Any(idx_df)
        str(col),
    Out:
        Series(real_claimants),
    Description:
        get the average fault in accidents
    '''
    # get paid loss by policy
    df_claim = df_claim.groupby(level=0).agg({'number_of_claimants': np.mean})
    # get paid loss by insured's id
    real_claimants = df_claim.loc[idx_df, 'number_of_claimants'].fillna(0)

    return(real_claimants)


def get_bs2_real_next_known(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df)
    Out:
        Series(real_next_known),
    Description:
        get known next premium
    '''
    df_policy = df_policy.groupby(level=0).agg({'Prior_Policy_Number': lambda x: x.iloc[0], 'Premium': np.sum})
    real_next_known = df_policy['Prior_Policy_Number'].map(df_policy['Premium'])

    return(real_next_known.loc[idx_df])