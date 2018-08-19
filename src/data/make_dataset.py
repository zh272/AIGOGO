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
    df = df_policy.groupby(level=0).agg({'Premium': np.nansum, col: lambda x: x.iloc[0]})
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
    '''

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
    if os.getcwd()[-1]=='O':
        raw_data_path = os.path.join(os.getcwd(), 'data', 'raw')
    else:
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
    if os.getcwd()[-1]=='O':
        interim_data_path = os.path.join(os.getcwd(), 'data', 'interim')
    else:
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
    if os.getcwd()[-1]=='O':
            interim_data_path = os.path.join(os.getcwd(), 'data', 'interim')
    else:
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

    #X_train, X_test, y_train = create_train_test_data_main_coverage(df_train, df_test, df_policy, df_claim, df_map_coverage)

    X_train, X_test, y_train = create_train_test_data_bs(df_train, df_test, df_policy, df_claim)

    write_test_data(X_train, "X_train_bs.csv")
    write_test_data(X_test, "X_test_bs.csv")
    write_test_data(y_train, "y_train_bs.csv")