import numpy as np
import os
import pandas as pd

def read_raw_data(file_name):
    '''
    In: file_name

    Out: raw_data

    Description: read data from directory /data/raw
    '''
    # set the path of raw data
    raw_data_path = os.path.join(os.path.pardir, os.path.pardir, 'data', 'raw')

    file_path = os.path.join(raw_data_path, file_name)
    raw_data = pd.read_csv(file_path, index_col='Policy_Number')

    return(raw_data)


def create_sample_data(df_train, df_claim, df_policy, data_size=2000):
    '''
    In:
        df_train, # contains dependent variable
        df_claim,
        df_policy,
        data_size, # sample size of df_train

    Out:
        df_sample

    Description:
        join method - aggregate claim and policy information by policy number
        limitation - lost information on claim and policy
    '''
    # sample train data
    df_sample = df_train.sample(n=data_size, random_state=0)

    # get aggregate function maps
    cols_polisum = ['Insured_Amount1', 'Insured_Amount2', 'Insured_Amount3', 'Premium']
    cols_polimix = ['Main_Insurance_Coverage_Group', 'Insurance_Coverage']
    map_policy = dict.fromkeys(df_policy.columns, lambda x: x.iloc[0])
    map_policy.update(dict.fromkeys(cols_polisum, np.sum))
    # select policies relevant to sample train data, aggregate at policy number level
    # insurance coverage info lost
    agg_policy = df_policy.loc[df_sample.index]
    agg_policy = agg_policy.groupby(level=0).agg(map_policy)
    agg_policy[cols_polimix] = 'Mix'

    # get aggregate function maps
    map_claim = {
                    'Paid_Loss_Amount': np.sum,
                    'paid_Expenses_Amount': np.sum,
                    'Claim_Number': lambda x: len(x[x.notnull()])
                }
    # select claims relevant to sample train data, aggregate at policy number level
    agg_claim = df_claim.loc[df_sample.index]
    agg_claim = agg_claim.groupby(level=0).agg(map_claim)

    # merge train, policy, and claim data
    df_sample = df_sample.merge(agg_policy, left_index=True, right_index=True)
    df_sample = df_sample.merge(agg_claim, left_index=True, right_index=True)

    return(df_sample)


def write_sample_data(df):
    '''
    In: df

    Out: None

    Description: write sample data to directory /data/interim
    '''
    interim_data_path = os.path.join(os.path.pardir, os.path.pardir, 'data', 'interim')
    write_sample_path = os.path.join(interim_data_path, 'sample-set.csv')
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
    #df_policy = read_raw_data('claim_0702.csv')

    df_sample = create_sample_data(df_train, df_claim, df_policy)
    write_sample_data(df_sample)