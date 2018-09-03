import os
import fire
import numpy as np
import pandas as pd
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

    return(df.loc[idx_df, col].fillna(0))


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

    return(real_prem_col.loc[idx_df].fillna(0))


def get_bs_real_prem_grp(df_policy, idx_df, grp):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df)
        str(col),
    Out:
        Series(real_prem_grp),
    Description:
        get premium by main coverage group
    '''
    df_policy = df_policy[df_policy['Main_Insurance_Coverage_Group']==grp]
    real_prem_grp = df_policy.groupby(level=0).agg({'Premium': np.nansum})

    return(real_prem_grp.loc[idx_df].fillna(0))


def get_bs_real_prem_exst(df_policy, idx_df, col, agg_prem):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df)
        str(col),
    Out:
        Series(real_prem_exst),
    Description:
        get feature average premium excluding terminated coverages
    '''
    # excluding terminated coverages
    real_ia = df_policy['Insured_Amount1'] + df_policy['Insured_Amount2'] + df_policy['Insured_Amount3']
    df_policy = df_policy[real_ia != 0]
    # get aggregated premium
    real_prem_col = agg_prem(df_policy, idx_df, col)

    return(real_prem_col)


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

    return(real_prem_ic.loc[idx_df, 'Premium'].fillna(0))


def get_bs_real_ic_indiv(df_policy, idx_df, ic, col):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(real_ic_indiv),
    Description:
        get information by insurance coverage
    '''
    # get insurance coverage rows
    df_ic = df_policy[df_policy['Insurance_Coverage']==ic]
    # get insured amount
    real_ia_ic_indiv = df_ic.loc[idx_df, col].fillna(0)

    return(real_ia_ic_indiv)


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


def get_bs_cat_ic_cluster(df_policy, idx_df, col, ncluster):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df)
        str(col),
    Out:
        Series(cat_ic_cluster),
    Description:
        get interaction of two categorical features
    '''
    # get cluster column
    df_col = df_policy.groupby(level=0).agg({col: lambda x: x.iloc[0]})
    # get premium by insurance coverage
    df_policy = df_policy.set_index(['Insurance_Coverage'], append=True)
    df_policy = df_policy['Premium'].unstack(level=1)
    df_policy.columns = [col[1] for col in df_policy.columns]
    # get premium by cluster column
    df_policy = df_policy.merge(df_col, how='left', left_index=True, right_index=True)
    df_col = df_policy.groupby([col]).mean().fillna(0)
    # get clusters
    mod_cluster = KMeans(n_clusters=ncluster)
    df_cluster = pd.Series(mod_cluster.fit_predict(df_col), index=df_col.index)
    cat_ic_cluster = df_policy[col].map(df_cluster)
    cat_ic_cluster = cat_ic_cluster.loc[idx_df].fillna(-1)

    return(cat_ic_cluster)


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
        real_mc_mean_diff = real_mc_mean_diff.where(~pd.isnull(real_mc_mean_diff), np.mean(y_train['real_mc_mean_diff'])) + X_valid['real_prem_plc']

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
    df_policy = df_policy.groupby(level=0).agg({'ibirth': lambda x: x.iloc[0]})
    df_claim = df_claim.groupby(level=0).agg({'DOB_of_Driver': lambda x: x.iloc[0], "Driver's_Relationship_with_Insured": lambda x: x.iloc[0]})
    df_claim = df_claim.merge(df_policy, how='left', left_index=True, right_index=True)
    cat_claim_ins = (df_claim["Driver's_Relationship_with_Insured"] == 1) & (df_claim['ibirth'] == df_claim['DOB_of_Driver'])

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
    real_loss_ins = real_loss_ins.loc[idx_df].fillna(-1)

    return(real_loss_ins)


def get_bs_real_prem_ic_nmf(df_policy, idx_df, method='nmf'):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        DataFrame(real_prem_ic_nmf),
    Description:
        get premium by insurance coverage, with nonnegative matrix factorization
    '''
    # remove terminated cols
    real_ia = df_policy['Insured_Amount1'] + df_policy['Insured_Amount2'] + df_policy['Insured_Amount3']
    df_policy = df_policy[real_ia != 0]
    # rows: policy number; cols: insurance coverage
    df_policy = df_policy.set_index('Insurance_Coverage', append=True)
    df_policy = df_policy[['Premium']].unstack(level=1)
    # transform dataframe to matrix
    mtx_df = df_policy.fillna(0).as_matrix()

    ### Uncomment below for creating csv file of 60 premium features
    # interim_data_path = os.path.join(os.path.dirname('__file__'), os.path.pardir, os.path.pardir, 'data', 'interim')
    # write_sample_path = os.path.join(interim_data_path, 'premium_60_new.csv')
    # df_policy.fillna(0).to_csv(write_sample_path)

    
    if method=='nn':
        import torch
        # nn dimension reduction
        model = torch.load(os.path.join('../models/saved_models', 'prem60_11.pt'))
        model.eval() # evaluation mode
        inp = torch.FloatTensor(mtx_df)
        with torch.no_grad():
            if torch.cuda.is_available():
                inp = torch.autograd.Variable(inp.cuda())
            else:
                inp = torch.autograd.Variable(inp)
        nmf_df = inp
        modulelist = list(model.regressor.modules())
        for l in modulelist[1:-1]:
            nmf_df = l(nmf_df)
        nmf_df = nmf_df.cpu().data.numpy()
    else:
        # non-negative matrix factorization
        nmf_df = NMF(n_components=7, random_state=1, alpha=.1, l1_ratio=.5).fit_transform(mtx_df)

    

    #
    n_comp = nmf_df.shape[1]
    print('>>> number of reduced features: {}'.format(n_comp))
    real_prem_ic_nmf = pd.DataFrame(nmf_df, index = df_policy.index).fillna(0)
    real_prem_ic_nmf.columns = ['real_prem_ic_nmf_' + str(i) for i in range(1, n_comp+1)]

    return(real_prem_ic_nmf.loc[idx_df].fillna(0))


def get_bs_real_ia_ic_nmf(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        DataFrame(real_ia_ic_nmf),
    Description:
        get issured amount by insurance coverage, with nonnegative matrix factorization
    '''
    # get scaled dominate insured amount by coverage
    values_ic = list(df_policy['Insurance_Coverage'].unique())
    list_ia = []
    for ic in values_ic:
        df_ic = df_policy[df_policy['Insurance_Coverage'] == ic]
        ia1 = list(df_ic['Insured_Amount1'].unique())
        ia2 = list(df_ic['Insured_Amount2'].unique())
        ia1_pos = [i for i in ia1 if i > 0]
        ia2_pos = [i for i in ia2 if i > 0]
        # use insured amount in the order of 1, 2, 3
        if (len(ia1_pos) > len(df_ic) / 2):
            ia = 'Insured_Amount1'
        elif (len(ia2_pos) > len(df_ic) / 2):
            ia = 'Insured_Amount2'
        else:
            ia = 'Insured_Amount3'
        # scale the insured amount in log term
        prem_mean = df_ic['Premium'].mean()
        ia_mean = (df_ic[ia] + 1).map(np.log).mean()
        ia_ic = (df_ic[ia] + 1).map(np.log) / ia_mean * prem_mean
        try:
            list_ia.append(ia_ic.loc[idx_df].fillna(0))
        except:
            list_ia.append(pd.Series(0, index=idx_df))
    ia_ic = pd.concat(list_ia, axis = 1)

    # transform dataframe to matrix
    mtx_df = ia_ic.fillna(0).as_matrix()
    # non-negative matrix factorization
    nmf_df = NMF(n_components=5, random_state=1, alpha=.1, l1_ratio=.5).fit_transform(mtx_df)
    # get insured amount
    real_ia_ic_nmf = pd.DataFrame(nmf_df, index = ia_ic.index)
    real_ia_ic_nmf.columns = ['real_ia_ic_nmf_' + str(i) for i in range(1, 6)]

    return(real_ia_ic_nmf.loc[idx_df].fillna(0))


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
    cat_ins_self = df_policy['ibirth'] == df_policy['dbirth']

    return(cat_ins_self.loc[idx_df])


def get_bs_cat_claim_theft(df_claim, idx_df):
    '''
    In:
        DataFrame(df_claim),
        Any(idx_df)
        str(col),
    Out:
        Series(cat_claim_theft),
    Description:
        get whether the vehicle was stolen
    '''
    ic_theft = ['05N', '09@', '09I', '10A', '68E', '68N']
    df_claim = df_claim.assign(cat_claim_theft = df_claim['Coverage'].map(lambda x: 1 if x in ic_theft else 0))
    df_claim = df_claim.groupby(level=0).agg({'cat_claim_theft': np.max})
    cat_claim_theft = df_claim.loc[idx_df, 'cat_claim_theft'].fillna(-1)

    return(cat_claim_theft)


def get_bs_cat_vequip(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df)
        str(col),
    Out:
        Series(cat_vequip),
    Description:
        get whether the vehicle has additional equipment
    '''
    # get whether the vehicle has additional equipment
    cols_vequip = ['fequipment' + str(i) for i in list(range(1, 10)) if i not in [7, 8]]
    cat_vequip = df_policy[cols_vequip].sum(axis=1).map(lambda x: 0 if x == 0 else 1)
    cat_vequip.columns = ['cat_vequip']
    # group by policy number
    cat_vequip = cat_vequip.groupby(level=0).agg({'cat_vequip': lambda x: x.iloc[0]})
    return(cat_vequip.loc[idx_df])


def get_bs_real_claim_fault(df_claim, idx_df):
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
    cat_claim_theft = df_claim.loc[idx_df, 'At_Fault?'].fillna(-1)

    return(cat_claim_theft)


def get_bs_cat_age(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(cat_age),
    Description:
        get inssured age
    '''
    df_policy = df_policy.groupby(level=0).agg({'ibirth': lambda x: x.iloc[0]})

    get_cat_age = lambda x: 0 if pd.isnull(x) else 2016 - int(x[3:])
    cat_age = df_policy['ibirth'].map(get_cat_age)
    cut_edge = [-1, 0, 20, 26, 31, 36, 39, 71, 75, 300]
    cat_age = pd.cut(cat_age, cut_edge, labels = list(range(1, len(cut_edge))))

    return(cat_age)


def get_bs_cat_ic_combo(df_policy, idx_df):
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
    df_policy = df_policy.groupby(level=0).agg({'Insurance_Coverage': lambda x: '|'.join(x[:3].sort_values())})
    cat_ic_combo = df_policy.loc[idx_df, 'Insurance_Coverage']

    return(cat_ic_combo)


def get_bs_real_prem_var_ic(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(real_prem_var_ic),
    Description:
        get premium variance by insured coverage
    '''
    # get map of premium to category interaction between ic and col
    df_map = df_policy.groupby(['Insurance_Coverage']).agg({'Premium': np.var})
    df_map.columns = ['var_ic']
    # map category interaction premium variance to policy number
    real_prem_var_ic = df_policy[['Insurance_Coverage', 'Premium']].merge(df_map, how='left', left_on=['Insurance_Coverage'], right_index=True)
    # get weighted average of variance on weight premium
    real_prem_var_ic['var_ic'] = real_prem_var_ic['var_ic'] * real_prem_var_ic['Premium']
    real_prem_var_ic = real_prem_var_ic.groupby(level=0).agg({'Premium': np.sum, 'var_ic': np.sum})
    real_prem_var_ic = real_prem_var_ic['var_ic'] / real_prem_var_ic['Premium']

    return(real_prem_var_ic.loc[idx_df])


######## get pre feature selection data set ########
def create_feature_selection_data(df_policy, df_claim, red_method='nmf'):
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
    X_train = read_interim_data('X_train_bs.csv')
    X_valid = read_interim_data('X_valid_bs.csv')
    X_test = read_interim_data('X_test_bs.csv')

    X_train = X_train.fillna(0)
    X_valid = X_valid.fillna(0)
    X_test = X_test.fillna(0)

    y_train = read_interim_data('y_train_bs.csv')
    y_valid = read_interim_data('y_valid_bs.csv')
    y_test = read_interim_data('y_test_bs.csv')

    X_fs = pd.concat([X_train, X_valid, X_test])
    y_fs = pd.concat([y_train, y_valid, y_test])

    # basic
#    print('Getting column cat_zip')
#    X_fs = X_fs.assign(cat_zip = get_bs_cat(df_policy, X_fs.index, 'aassured_zip'))
#    print('Getting column cat_ins_self')
#    X_fs = X_fs.assign(cat_ins_self = get_bs_cat_ins_self(df_policy, X_fs.index))
#
#    # distribution
#    print('Getting column real_prem_ic_distr')
#    X_fs = X_fs.assign(real_prem_ic_distr = get_bs_real_prem_ic(df_policy, X_fs.index, 'Distribution_Channel'))
#
    # insurance coverage
    print('Getting column real_prem_16G_indiv')
    X_fs = X_fs.assign(real_prem_16G_indiv = get_bs_real_ic_indiv(df_policy, X_fs.index, '16G', 'Premium'))

    print('Getting column real_ia1_16G_indiv')
    X_fs = X_fs.assign(real_ia1_16G_indiv = get_bs_real_ic_indiv(df_policy, X_fs.index, '16G', 'Insured_Amount1'))
#    # vehicle
#    print('Getting column real_prem_ic_vmy')
#    X_fs = X_fs.assign(real_prem_ic_vmy = get_bs_real_prem_exst(df_policy, X_fs.index, 'Manafactured_Year_and_Month', get_bs_real_prem_ic))
#
#    print('Getting column real_prem_per_vcost')
#    X_fs = X_fs.assign(real_prem_per_vcost = X_fs['real_prem_plc'] / X_fs['real_vcost'])
#
    print('Getting column cat_vequip')
    X_fs = X_fs.assign(cat_vequip = get_bs_cat_vequip(df_policy, X_fs.index))

#    print('Getting column cat_ic_cluster_vmm2')
#    X_fs = X_fs.assign(cat_ic_cluster_vmm2 = get_bs_cat_ic_cluster(df_policy, X_fs.index, 'Vehicle_Make_and_Model2', 60))

    # claim
    print('Getting column cat_claim_ins')
    X_fs = X_fs.assign(cat_claim_ins = get_bs_cat_claim_ins(df_policy, df_claim, X_fs.index))

    print('Getting column real_loss_ins')
    X_fs = X_fs.assign(real_loss_ins = get_bs_real_loss_ins(df_policy, df_claim, X_fs.index))

    print('Getting column cat_claim_theft')
    X_fs = X_fs.assign(cat_claim_theft = get_bs_cat_claim_theft(df_claim, X_fs.index))
#
#    print('Getting column real_claim_fault')
#    X_fs = X_fs.assign(real_claim_fault = get_bs_real_claim_fault(df_claim, X_fs.index))

    # insurance coverage
    print('Getting column real_prem_ic_nmf')
    temp = get_bs_real_prem_ic_nmf(df_policy, X_fs.index, method=red_method)
    n_comp = temp.shape[1]
    print('>>> number of reduced features: {}'.format(n_comp))
    colnames = ['real_prem_ic_nmf_' + str(i) for i in range(1, n_comp+1)]
    X_fs[colnames] = temp

#    print('Getting column real_ia_ic_nmf')
#    colnames = ['real_ia_ic_nmf_' + str(i) for i in range(1, 6)]
#    X_fs[colnames] = get_bs_real_ia_ic_nmf(df_policy, X_fs.index)

    print('Getting column real_prem_terminate')
    X_fs = X_fs.assign(real_prem_terminate = get_bs_real_prem_terminate(df_policy, X_fs.index))

    print('Getting column real_prem_dmg')
    X_fs = X_fs.assign(real_prem_dmg = get_bs_real_prem_exst(df_policy, X_fs.index, '車損', get_bs_real_prem_grp))

    print('Getting column real_prem_lia')
    X_fs = X_fs.assign(real_prem_lia = get_bs_real_prem_exst(df_policy, X_fs.index, '車責', get_bs_real_prem_grp))

    print('Getting column real_prem_thf')
    X_fs = X_fs.assign(real_prem_thf = get_bs_real_prem_exst(df_policy, X_fs.index, '竊盜', get_bs_real_prem_grp))

    print('Getting column real_prem_ic')
    X_fs = X_fs.assign(real_prem_ic = get_bs_real_prem_exst(df_policy, X_fs.index, 'Main_Insurance_Coverage_Group', get_bs_real_prem_ic))

    print('Getting column real_prem_var_ic')
    X_fs = X_fs.assign(real_prem_var_ic = get_bs_real_prem_var_ic(df_policy, X_fs.index))

    print('Getting column cat_ic_combo')
    X_fs = X_fs.assign(cat_ic_combo = get_bs_cat_ic_combo(df_policy, X_fs.index))

    # feature template expansion
    cols_cat = [col for col in X_fs.columns if col.startswith('cat') and col not in X_train.columns]
    cols_cat_all = [col for col in X_fs.columns if col.startswith('cat')]

    # frequency of category values
    for col_cat in cols_cat:
        col_freq = col_cat.replace('cat_', 'real_freq_')
        print('Getting column ' + col_freq)
        X_fs[col_freq] = get_bs_real_freq(X_fs, X_fs.index, col_cat)

    # train valid test split
    X_train = X_fs.loc[X_train.index]
    X_valid = X_fs.loc[X_valid.index]
    X_train_all = pd.concat([X_train, X_valid])
    X_test = X_fs.loc[X_test.index]

    y_train = y_fs.loc[y_train.index]
    y_valid = y_fs.loc[y_valid.index]
    y_train_all = pd.concat([y_train, y_valid])
    y_test = y_fs.loc[y_test.index]


    # add mean encoding
    # add mean encoding on next_premium mean
    for col_cat in cols_cat:
        col_mean = col_cat.replace('cat_', 'real_mc_mean_')
        print('Getting column ' + col_mean)
        X_test[col_mean] = get_bs_real_mc_mean(col_cat, X_train, y_train, X_valid=X_test, train_only=False, fold=5, prior=1000)
        X_valid[col_mean] = get_bs_real_mc_mean(col_cat, X_train, y_train, X_valid=X_valid, train_only=False, fold=5, prior=1000)
        X_train[col_mean] = get_bs_real_mc_mean(col_cat, X_train, y_train, X_valid=pd.DataFrame(), train_only=True, fold=5, prior=1000)
        X_train_all[col_mean] = get_bs_real_mc_mean(col_cat, X_train_all, y_train_all, X_valid=pd.DataFrame(), train_only=True, fold=5, prior=1000)

    # add mean encoding on mean of diff btw next_premium and premium
    for col_cat in cols_cat_all:
        col_mean = col_cat.replace('cat_', 'real_mc_mean_diff_')
        print('Getting column ' + col_mean)
        X_test[col_mean] = get_bs_real_mc_mean_diff(col_cat, X_train, y_train, X_valid=X_test, train_only=False, fold=5, prior=1000)
        X_valid[col_mean] = get_bs_real_mc_mean_diff(col_cat, X_train, y_train, X_valid=X_valid, train_only=False, fold=5, prior=1000)
        X_train[col_mean] = get_bs_real_mc_mean_diff(col_cat, X_train, y_train, X_valid=pd.DataFrame(), train_only=True, fold=5, prior=1000)

    # add mean encoding on probability of next_premium being 0
    for col_cat in cols_cat:
        col_prob = col_cat.replace('cat_', 'real_mc_prob_')
        print('Getting column ' + col_prob)
        X_test[col_prob] = get_bs_real_mc_prob(col_cat, X_train, y_train, X_valid=X_test, train_only=False, fold=5, prior=1000)
        X_valid[col_prob] = get_bs_real_mc_prob(col_cat, X_train, y_train, X_valid=X_valid, train_only=False, fold=5, prior=1000)
        X_train[col_prob] = get_bs_real_mc_prob(col_cat, X_train, y_train, X_valid=pd.DataFrame(), train_only=True, fold=5, prior=1000)
        X_train_all[col_prob] = get_bs_real_mc_prob(col_cat, X_train_all, y_train_all, X_valid=pd.DataFrame(), train_only=True, fold=5, prior=1000)

    write_test_data(X_train, "X_train_prefs.csv")
    write_test_data(y_train, "y_train_prefs.csv")
    write_test_data(X_valid, "X_valid_prefs.csv")
    write_test_data(y_valid, "y_valid_prefs.csv")
    write_test_data(X_train_all, "X_train_all_prefs.csv")
    write_test_data(y_train_all, "y_train_all_prefs.csv")
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

    cols_ex = []
    cols_train = [col for col in X_train.columns if not col.startswith('cat') and col not in cols_ex]

    All_train = y_train.merge(X_train, how='left', left_index=True, right_index=True)
    All_valid = y_valid.merge(X_valid, how='left', left_index=True, right_index=True)

    lgb_train = lgb.Dataset(All_train[cols_train].values, All_train['Next_Premium'].values.flatten(), free_raw_data=False)
    lgb_valid = lgb.Dataset(All_valid[cols_train].values, All_valid['Next_Premium'].values.flatten(), reference=lgb_train, free_raw_data=False)

    model = lgb.train(
        params['model'], lgb_train, valid_sets=lgb_valid, **params['train']
    )

    valid_pred = model.predict(All_valid[cols_train])
    valid_pred = pd.DataFrame(valid_pred, index = All_valid.index)
    valid_pred.columns = ['Next_Premium']
    write_test_data(valid_pred, 'fit_valid_prefs.csv')

    train_pred = model.predict(All_train[cols_train])
    train_pred = pd.DataFrame(train_pred, index = All_train.index)
    train_pred.columns = ['Next_Premium']
    write_test_data(train_pred, 'fit_train_prefs.csv')

    valid_mae = mean_absolute_error(All_valid['Next_Premium'], valid_pred)
    print('pre-selection mae is {}'.format(valid_mae))

    varimp = list(model.feature_importance())
    varimp = dict(zip(cols_train, varimp))
    for key, value in sorted(varimp.items(), key=lambda x: -x[1]):
        print("%s: %s" % (key, value))

    lgb_output = {'varimp': varimp, 'mae': valid_mae}

    return(lgb_output)


def get_bs_quick_submission(params):
    '''
    In:

    Out:
        float(mae)

    Description:
        calculate quick mae on validation set
    '''
    X_train = read_interim_data('X_train_all_prefs.csv')
    y_train = read_interim_data('y_train_all_prefs.csv')
    X_valid = read_interim_data('X_test_prefs.csv')
    y_valid = read_interim_data('y_test_prefs.csv')

    # preprocessing
    X_train.fillna(-999, inplace=True)
    X_valid.fillna(-999, inplace=True)

    cols_train = [col for col in X_train.columns if not col.startswith('cat')]

    All_train = y_train.merge(X_train, how='left', left_index=True, right_index=True)
    All_valid = y_valid.merge(X_valid, how='left', left_index=True, right_index=True)

    lgb_train = lgb.Dataset(All_train[cols_train].values, All_train['Next_Premium'].values.flatten(), free_raw_data=False)

    model = lgb.train(
        params['model'], lgb_train, **params['train']
    )

    valid_pred = model.predict(All_valid[cols_train])
    valid_pred = pd.DataFrame(valid_pred, index = All_valid.index)
    valid_pred.columns = ['Next_Premium']
    write_test_data(valid_pred, 'testing-set.csv')

    return(None)


######## read/write func ########
def read_raw_data(file_name, index_col='Policy_Number'):
    '''
    In: file_name

    Out: raw_data

    Description: read data from directory /data/raw
    '''
    # set the path of raw data
    if os.getcwd()[-1]=='O':
        raw_data_path = os.path.join(os.path.dirname('__file__'), 'data', 'raw') #os.getcwd(), should direct to the path /AIGOGO
    else: #os.getcwd()[-1]=='a':
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
    if os.getcwd()[-1]=='O':
        interim_data_path = os.path.join(os.path.dirname('__file__'), 'data', 'interim') #os.getcwd(), should direct to the path /AIGOGO
    else: #os.getcwd()[-1]=='a':
        interim_data_path = os.path.join(os.path.dirname('__file__'), os.path.pardir, os.path.pardir, 'data', 'interim')

    file_path = os.path.join(interim_data_path, file_name)
    interim_data = pd.read_csv(file_path, index_col=index_col)

    return(interim_data)


def write_test_data(df, file_name, red_method='nmf'):
    '''
    In:
        DataFrame(df),
        str(file_name),
    Out:
        None

    Description:
        Write sample data to directory /data/interim
    '''
    if os.getcwd()[-1]=='O':
        interim_data_path = os.path.join(os.path.dirname('__file__'), 'data', 'interim') #os.getcwd(), should direct to the path /AIGOGO
    else: #os.getcwd()[-1]=='a':
        interim_data_path = os.path.join(os.path.dirname('__file__'), os.path.pardir, os.path.pardir, 'data', 'interim')

    write_sample_path = os.path.join(interim_data_path, file_name)
    df.to_csv(write_sample_path)

    return(None)

def demo(red_method='nmf'):
    '''
    train data: training-set.csv
    test data: testing-set.csv
    independent_claim: claim_0702.csv
    independent_policy: policy_0702.csv
    '''
    
    # df_train = read_raw_data('training-set.csv')
    # df_test = read_raw_data('testing-set.csv')
    df_claim = read_raw_data('claim_0702.csv')
    df_policy = read_raw_data('policy_0702.csv')

    create_feature_selection_data(df_policy, df_claim, red_method=red_method)

    lgb_model_params = {
        'boosting_type': 'gbdt',
        'num_iterations': 5000,
        'max_depth':-1,
        'objective': 'regression_l1',
        'metric': 'mae',
        'lamba_l1':0.1,
        'num_leaves': 40,
        'learning_rate': 0.01,
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'subsample_freq': 5,
        'min_data_in_leaf': 15,
        'min_gain_to_split': 0.01,
        'seed': 0,
    }
    lgb_train_params = {
        'early_stopping_rounds': 3,
        'learning_rates': None, # lambda iter: 0.1*(0.99**iter),
        'verbose_eval': True,
    }
    lgb_params = {'model': lgb_model_params, 'train': lgb_train_params}
    lgb_output = get_bs_quick_mae(lgb_params)
#    lgb_output = get_bs_quick_submission(lgb_params)


if __name__ == '__main__':
    fire.Fire(demo)