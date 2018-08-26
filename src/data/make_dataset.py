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


######## manual feature ########
def get_bs_cat_age(df_policy, idx_df):
    '''
    In:
        DataFrame(df_policy),
        Any(idx_df),
    Out:
        Series(cat_vmy),
    Description:
        get inssured
    '''
    df_policy = df_policy.groupby(level=0).agg({'ibirth': lambda x: x.iloc[0]})
    def get_cat_age(x):
        cat_age = 0
        if not pd.isnull(x):
            cat_age = 2016 - int(x[3:])
        if cat_age <= 22 and cat_age > 0:
            cat_age = -1
        if cat_age >= 82:
            cat_age = -2
        return(cat_age)
    cat_age = df_policy['ibirth'].map(get_cat_age)

    return(cat_age)


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

#    print('Getting column cat_age')
#    df_bs = df_bs.assign(cat_age = get_bs_cat_age(df_policy, df_bs.index))

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

    X_train, X_test, y_train = create_train_test_data_bs(df_train, df_test, df_policy, df_claim)

    write_test_data(X_train, "X_train_bs.csv")
    write_test_data(X_test, "X_test_bs.csv")
    write_test_data(y_train, "y_train_bs.csv")