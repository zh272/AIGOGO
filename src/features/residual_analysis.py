import numpy as np
import pandas as pd
import os

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


def get_analysis_table(fitted, feature):
    '''
    In:
        DataFrame(fitted),
        str(col),

    Out:
        DataFrame(analysis)

    Description:
        give analysis of fitting error on column
    '''
    fitted = fitted.merge(feature.to_frame(name='col'), how='left', left_index=True, right_index=True)
    analysis = fitted.groupby(['col']).agg({'error': [np.mean, np.std, np.size]})

    return(analysis)

######## feature analysis func ########
def get_anal_age(fitted, df_policy):
    '''
    In:
        DataFrame(fitted),
        DataFrame(df_policy),

    Out:
        DataFrame(anal_age)

    Description:
        Analyze fitting error by age group
    '''
    # get age feature
    df_policy = df_policy.groupby(level=0).agg({'ibirth': lambda x: x.iloc[0]})
    get_cat_age = lambda x: 0 if pd.isnull(x) else (2016 - int(x[3:]))
    cat_age = df_policy['ibirth'].map(get_cat_age)

    # aggrage error by age
    anal_age = get_analysis_table(fitted, cat_age)

    return(anal_age)


def get_anal_assured(fitted, df_policy):
    '''
    In:
        DataFrame(fitted),
        DataFrame(df_policy),

    Out:
        DataFrame(anal_assured)

    Description:
        Analyze fitting error by assured group
    '''
    # get age feature
    cat_assured = get_bs_cat(df_policy, fitted.index, 'fassured')

    # aggrage error by age
    anal_assured = get_analysis_table(fitted, cat_assured)

    return(anal_assured)


def get_anal_zip(fitted, df_policy):
    '''
    In:
        DataFrame(fitted),
        DataFrame(df_policy),

    Out:
        DataFrame(anal_zip)

    Description:
        Analyze fitting error by assured zip
    '''
    # get age feature
    cat_zip = get_bs_cat(df_policy, fitted.index, 'aassured_zip')

    # aggrage error by age
    anal_zip = get_analysis_table(fitted, cat_zip)

    return(anal_zip)


######## residual analysis func ########
def get_residual_analysis(label, fitted, df_policy, df_claim):
    '''
    In:
        DataFrame(label),
        DataFrame(fitted),
        DataFrame(df_policy),
        DataFrame(df_claim),

    Out:
        dict(results) -> results contain analysis statistics

    Description:
        Analyze valid fitted results on existing features, and analyze training fitted results on left out features
    '''
    # get fitting results
    fitted = label.merge(fitted, how='left', left_index=True, right_index=True)
    fitted.columns = ['label', 'fitted']
    fitted = fitted.assign(error = fitted['label'] - fitted['fitted'])

    print('Analyzing cat_age')
    anal_age = get_anal_age(fitted, df_policy)

    print('Analyzing cat_assured')
    anal_assured = get_anal_assured(fitted, df_policy)

    print('Analyzing cat_zip')
    anal_zip = get_anal_zip(fitted, df_policy)

    results = {'anal_age': anal_age,
               'anal_assured': anal_assured,
               'anal_zip': anal_zip,
               }

    return(results)

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
    df_train_label = read_interim_data('y_train_prefs.csv')
    df_train_fit = read_interim_data('fit_train_prefs.csv')
    df_valid_label = read_interim_data('y_valid_prefs.csv')
    df_valid_fit = read_interim_data('fit_valid_prefs.csv')

    results = get_residual_analysis(df_train_label, df_train_fit, df_policy, df_claim)

