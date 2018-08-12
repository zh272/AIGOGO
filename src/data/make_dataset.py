import numpy as np
import os
import pandas as pd
import h2o
import matplotlib.pyplot as plt
from h2o.estimators.random_forest import H2ORandomForestEstimator

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
    df = df_policy.groupby(level=0).agg({'Premium': np.nansum})
    df = df.groupby([col]).agg({'Premium': np.nanmedian})
    df = df_policy[[col]].merge(df, how='left', left_on=[col], right_index=True)
    df = df.groupby(level=0).agg({'Premium': lambda x: x.iloc[0]})
    df = df.assign(real_prem = df['Premium'].map(lambda x: 0 if pd.isnull(x) else x))
    return(df.loc[idx_df, 'real_prem'])

######## column generate functions ########
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
    df = df.assign(cat_marriage = df['fsex'].map(lambda x: 0 if pd.isnull(x) else x))
    return(df['fsex'][idx_df])


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


######## get extention columns ########
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
    df = pd.qcut(df['Engine_Displacement_(Cubic_Centimeter)'], 10)
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
    df = df_policy['Imported_or_Domestic_Car'].map(lambda x: 1 if x == 10 else 0)
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
    cat_vmotor = np.where((df['qpt'] <= 3) & (df['qpt'] == 'P'), 2, 1)
    cat_vmotor = np.where(df['qpt'] != 'P', 0, cat_vmotor)
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
    df = df_policy.assign(cat_sex = get_bs_cat_sex(df_policy, df_policy.index))
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


def get_bs_real_prem_ic_distr(df_policy, df_claim, idx_df):
    '''
    In:
        DataFrame(df_policy),
        DataFrame(df_claim),
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


def get_bs_real_prem_area_distr(df_policy, df_claim, idx_df):
    '''
    In:
        DataFrame(df_policy),
        DataFrame(df_claim),
        Any(idx_df),
    Out:
        Series(real_prem_area_distr),
    Description:
        get premium by distribution channel * insurance coverage
    '''
    df = df_policy.groupby(['Insurance_Coverage', "Distribution_Channel"]).agg({'Premium': np.nanmedian})
    df = df_policy[['Insurance_Coverage', "Distribution_Channel"]].merge(df, how='left', left_on=['Insurance_Coverage', "Distribution_Channel"], right_index=True)
    df = df.groupby(level=0).agg({'Premium': np.nansum})
    df = df.assign(real_prem_ic_distr = df['Premium'].map(lambda x: 0 if pd.isnull(x) else x))
    return(df['real_prem_ic_distr'][idx_df])


def get_bs_real_freq_distr(df_policy, df_claim, idx_df):
    '''
    In:
        DataFrame(df_policy),
        DataFrame(df_claim),
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


######## get feature selection ########
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
    X_bs = read_interim_data('X_train_bs.csv')
    y_bs = read_interim_data('y_train_bs.csv')

    X_fs = X_bs
    y_fs = y_bs

    print('Getting column real_prem_vmm1')
    X_fs = X_fs.assign(real_prem_vmm1 = get_bs_real_prem_vmm1(df_policy, X_fs.index))

    write_test_data(X_train, "X_train_fs.csv")
    write_test_data(y_train, "y_train_fs.csv")

    return(X_fs, y_fs)


def stepwise_policy_feature_selection(df_policy, df_claim, dict_add=dict(), max_rounds=20, num_only=False):
    '''
    In:
        DataFrame(df_policy),
        DataFrame(df_claim),
        dict(dict_add), # add features on the fly
        float(max_rounds),
        bool(num_only), # include numeric features only
    Out:
        dict(output), # test mae and selected columns
    Description:
        do forward feature selection with penalty on additional freedom
    '''
    X_all = read_interim_data('X_train_fs.csv')
    y_all = read_interim_data('y_train_fs.csv')
    All_all = y_all.merge(X_all, how='left', left_index=True, right_index=True)

    # add new feature
    for col, func in dict_add.items():
        print('Getting column {}'.format(col))
        All_all[col] = func(df_policy, df_claim, All_all.index)

    # train test split
    np.random.seed(0)
    msk = np.random.rand(len(All_all)) < 0.8
    All_train = All_all[msk]
    All_test = All_all[~msk]

    # get schema
    schema = dict()
    for col in All_train.columns:
        if col.startswith('cat'):
            schema[col] = 'enum'
        elif col.startswith('int'):
            schema[col] = 'int'
        else:
            schema[col] = 'real'

    # get valid mae
    h2o.init(max_mem_size = "2G")
    def get_mae(cols_train, All_train=All_train, All_test=pd.DataFrame(), schema=schema):
        # clean memory
        h2o.remove_all()
        # get random forest model
        model = H2ORandomForestEstimator(ntrees = 30,
                                          max_depth = 15,
                                          stopping_metric = 'mae',
                                          stopping_rounds = 2,
                                          seed = 1000000
                                          )

        # tranform to h2o dataset
        h2o_train = h2o.H2OFrame(All_train, column_types=schema)
        if len(All_test) == 0:
            train, valid = h2o_train.split_frame(ratios = [0.8], seed=0)
        else:
            h2o_test = h2o.H2OFrame(All_test, column_types=schema)
            train, valid = (h2o_train, h2o_test)

        model.train(cols_train, 'Next_Premium', training_frame=train, validation_frame=valid)

        mae = model.mae(valid=True)
        return(mae)

    # initialize add delete columns
    cols_del = ['real_prem_plc', 'real_prem_ic_distr', 'real_prem_lia', 'cat_distr', 'int_acc_lia', 'cat_zip', 'cat_sex', 'real_acc_dmg', 'cat_vmy']
    cols_add = [col for col in X_all.columns if col not in cols_del] + list(dict_add.keys())

    if num_only:
        cols_add = [col for col in cols_add if not col.startswith('cat')]

    # stepwise select features
    col_change = 'init'
    rounds = 0
    print('sth')
    mae_min = get_mae(cols_del)
    print('Baseline data gives minimum mae {}'.format(mae_min))
    while(col_change and rounds < max_rounds):
        # initialize
        rounds = rounds + 1
        mae_lag = mae_min
        panelty_add = len(cols_add) / 10
        # backward step
        '''
        for col_del in cols_del:
            print('Testing delete column {}'.format(col_del))
            cols_train = [col for col in cols_del if col != col_del]
            mae = get_mae(cols_train)
            print('MAE changes to {}'.format(mae))
            col_change = col_change if mae >= mae_min else col_del
            mae_min = mae_min if mae >= mae_min else mae
        '''
        # forward step
        for col_add in cols_add:
            print('Testing add column {}'.format(col_add))
            cols_train = cols_del + [col_add]
            mae = get_mae(cols_train)
            print('MAE changes to {}'.format(mae))
            col_change = col_change if mae >= mae_min else col_add
            mae_min = mae_min if mae >= mae_min else mae

        if col_change in cols_add:
            print('Round {} adds column {}'.format(rounds, col_change))
            cols_del = cols_del + [col_change]
            cols_add = [col for col in cols_add if col != col_change]
        elif col_change in cols_del:
            print('Round {} removes column {}'.format(rounds, col_change))
            cols_add = cols_add + [col_change]
            cols_del = [col for col in cols_del if col != col_change]

        if mae_lag <= mae_min + panelty_add:
            col_change = None

        print('Existing {} columns are: {}'.format(len(cols_del), ', '.join(cols_del)))
        print('Round {} minimum mae is {}'.format(rounds, mae_min))

    result = {'mae': get_mae(cols_del, All_test=All_test),
              'cols': cols_del}

    h2o.shutdown(prompt=False)

    return(result)


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
    df_bs = pd.concat([df_train, df_test])

    print('Getting column cat_age')
    df_bs = df_bs.assign(cat_age = get_bs_cat_age(df_policy, df_bs.index))

    print('Getting column cat_area')
    df_bs = df_bs.assign(cat_area = get_bs_cat_area(df_policy, df_bs.index))

    print('Getting column cat_assured')
    df_bs = df_bs.assign(cat_assured = get_bs_cat_assured(df_policy, df_bs.index))

    print('Getting column cat_cancel')
    df_bs = df_bs.assign(cat_cancel = get_bs_cat_cancel(df_policy, df_bs.index))

    print('Getting column cat_distr')
    df_bs = df_bs.assign(cat_distr = get_bs_cat_distr(df_policy, df_bs.index))

    print('Getting column cat_marriage')
    df_bs = df_bs.assign(cat_marriage = get_bs_cat_marriage(df_policy, df_bs.index))

    print('Getting column cat_sex')
    df_bs = df_bs.assign(cat_sex = get_bs_cat_sex (df_policy, df_bs.index))

    print('Getting column cat_vc')
    df_bs = df_bs.assign(cat_vc = get_bs_cat_vc(df_policy, df_bs.index))

    print('Getting column cat_vmm1')
    df_bs = df_bs.assign(cat_vmm1 = get_bs_cat_vmm1(df_policy, df_bs.index))

    print('Getting column cat_vmm2')
    df_bs = df_bs.assign(cat_vmm2 = get_bs_cat_vmm2(df_policy, df_bs.index))

    print('Getting column cat_vmy')
    df_bs = df_bs.assign(cat_vmy = get_bs_cat_vmy(df_policy, df_bs.index))

    print('Getting column cat_vqpt')
    df_bs = df_bs.assign(cat_vqpt = get_bs_cat_vqpt(df_policy, df_bs.index))

    print('Getting column cat_vregion')
    df_bs = df_bs.assign(cat_vregion = get_bs_cat_vregion(df_policy, df_bs.index))

    print('Getting column cat_zip')
    df_bs = df_bs.assign(cat_zip = get_bs_cat_zip(df_policy, df_bs.index))

    print('Getting column int_acc_lia')
    df_bs = df_bs.assign(int_acc_lia = get_bs_int_acc_lia(df_policy, df_bs.index))

    print('Getting column int_claim')
    df_bs = df_bs.assign(int_claim_plc = get_bs_int_claim_plc(df_policy, df_claim, df_bs.index))

    print('Getting column int_others')
    df_bs = df_bs.assign(int_others = get_bs_int_others(df_policy, df_bs.index))

    print('Getting column real_acc_dmg')
    df_bs = df_bs.assign(real_acc_dmg = get_bs_real_acc_dmg(df_policy, df_bs.index))

    print('Getting column real_acc_lia')
    df_bs = df_bs.assign(real_acc_lia = get_bs_real_acc_lia(df_policy, df_bs.index))

    print('Getting column real_loss')
    df_bs = df_bs.assign(real_loss_plc = get_bs_real_loss_plc(df_policy, df_claim, df_bs.index))

    print('Getting column real_prem_dmg')
    df_bs = df_bs.assign(real_prem_dmg = get_bs_real_prem_dmg(df_policy, df_bs.index))

    print('Getting column real_prem_ins')
    df_bs = df_bs.assign(real_prem_ins = get_bs_real_prem_ins(df_policy, df_bs.index))

    print('Getting column real_prem_lia')
    df_bs = df_bs.assign(real_prem_lia = get_bs_real_prem_lia(df_policy, df_bs.index))

    print('Getting column real_prem_plc')
    df_bs = df_bs.assign(real_prem_plc = get_bs_real_prem_plc(df_policy, df_bs.index))

    print('Getting column real_prem_thf')
    df_bs = df_bs.assign(real_prem_thf = get_bs_real_prem_thf(df_policy, df_bs.index))

    print('Getting column real_prem_vc')
    df_bs = df_bs.assign(real_prem_vc = get_bs_real_prem_vc(df_policy, df_bs.index))

    print('Getting column real_vcost')
    df_bs = df_bs.assign(real_vcost = get_bs_real_vcost(df_policy, df_bs.index))

    print('Getting column real_ved')
    df_bs = df_bs.assign(real_ved = get_bs_real_ved(df_policy, df_bs.index))
    '''
    X_train = read_interim_data('X_train_bs.csv')
    X_test = read_interim_data('X_test_bs.csv')
    df_bs = pd.concat([X_train, X_test])

    # feature transformation
    print('Getting column real_prem_distr')
    df_bs = df_bs.drop(['cat_distr'], axis=1)
    #df_bs = df_bs.assign(real_prem_distr = get_bs_real_prem_distr(df_policy, df_bs.index))
    df_bs = df_bs.assign(real_prem_ic_distr = get_bs_real_prem_ic_distr(df_policy, df_bs.index))

    col_y = 'Next_Premium'
    cols_X = [col for col in df_bs.columns if col != col_y]

    X_train = df_bs.loc[df_train.index, cols_X]
    X_test = df_bs.loc[df_test.index, cols_X]
    y_train = df_train.loc[X_train.index]

    return(X_train, X_test, y_train)


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

    #df_train = read_raw_data('training-set.csv')
    #df_test = read_raw_data('testing-set.csv')
    #df_claim = read_raw_data('claim_0702.csv')
    #df_policy = read_raw_data('policy_0702.csv')

    #X_train, X_test, y_train = create_train_test_data_main_coverage(df_train, df_test, df_policy, df_claim, df_map_coverage)

    X_train, X_test, y_train = create_train_test_data_bs(df_train, df_test, df_policy, df_claim)

    write_test_data(X_train, "X_train_bs_ext.csv")
    write_test_data(X_test, "X_test_bs_ext.csv")
    write_test_data(y_train, "y_train_bs_ext.csv")