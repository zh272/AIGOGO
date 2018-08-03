import numpy as np
import os
import pandas as pd

def read_raw_data(file_name, index_col='Policy_Number'):
    '''
    In: file_name

    Out: raw_data

    Description: read data from directory /data/raw
    '''
    # set the path of raw data
    raw_data_path = os.path.join(os.path.pardir, os.path.pardir, 'data', 'raw')

    file_path = os.path.join(raw_data_path, file_name)
    raw_data = pd.read_csv(file_path, index_col=index_col)

    return(raw_data)


def get_main_coverage_aggregated_policy(df_policy):
    '''
    In:
        DataFrame(df_policy),
    Out:
        DataFrame(df_sample),
    Description:
        Aggregate policy infos on policy id using folloing methods:
            (1) for common information, keep the first row
            (2) sum up ['Insured_Amount1', 'Insured_Amount2', 'Insured_Amount3', 'Premium'] by 'Main_Insurance_Coverage_Group'
            (3) count 'Insurance_Coverage' by 'Main_Insurance_Coverage_Group'
            (4) label whether any 'Coverage_Deductible_if_applied' is positive by 'Main_Insurance_Coverage_Group'
    '''
    # aggregate coverage on Policy_Number
        # get columns per coverage
    col_coverage = ['Main_Insurance_Coverage_Group', 'Insurance_Coverage', 'Insured_Amount1', 'Insured_Amount2', 'Insured_Amount3', 'Coverage_Deductible_if_applied', 'Premium']
        # group coverage by Main_Insurance_Coverage_Group
    df_coverage = df_policy[col_coverage]
    df_coverage['Main_Insurance_Coverage_Group'] = df_coverage['Main_Insurance_Coverage_Group'].map({'車損': 'Damage', '竊盜': 'Theft', '車責': 'Liability'})
    df_coverage = df_coverage.set_index(['Main_Insurance_Coverage_Group'], append = True)
        # aggegate coverage items by sum
    key_coverage = [col for col in col_coverage if col != 'Main_Insurance_Coverage_Group']
    map_coverage = dict.fromkeys(key_coverage, np.sum)
    map_coverage['Insurance_Coverage'] = lambda x: len(x)
    map_coverage['Coverage_Deductible_if_applied'] = lambda x: x.sum() > 0
    agg_coverage = df_coverage.groupby(level=[0, 1]).agg(map_coverage)
    agg_coverage = agg_coverage.unstack(level=1)

    # each policy id corresponds to a unique row of policy columns
        # get columns per policy
    col_policy = [col for col in df_policy.columns if col not in col_coverage]
            # get policy info from first row, all rows should give the same info
    map_policy = dict.fromkeys(col_policy, lambda x: x.iloc[0])
    agg_policy = df_policy.groupby(level=0).agg(map_policy)

    # merge coverage info and policy info
    agg_policy = agg_policy.merge(agg_coverage, left_index=True, right_index=True)

    return(agg_policy)


def get_main_coverage_aggregated_claim(df_claim, df_map_coverage):
    '''
    In:
        DataFrame(df_claim),
        DataFrame(df_map_coverage), # maps coverage to main coverage group
    Out:
        DataFrame(agg_Claim),
    Description:
        Aggregate claim infos on policy id using folloing methods:
            (1) sum up ['Paid_Loss_Amount', 'paid_Expenses_Amount', 'Salvage_or_Subrogation?', 'Deductible', 'number_of_claimants'] by 'Main_Insurance_Coverage_Group'
            (2) count 'Claim_Number' by 'Main_Insurance_Coverage_Group'
            (3) average 'At_Fault?' by 'Main_Insurance_Coverage_Group'
    '''
    # get Main_Insurance_Coverage_Group from Coverage
    df_claim['Main_Insurance_Coverage_Group'] =  df_claim['Coverage'].map(df_map_coverage['Main_Insurance_Coverage_Group'])

    # aggregate claim by Main_Insurance_Coverage_Group to
    df_claim = df_claim.set_index(['Main_Insurance_Coverage_Group'], append = True)
    # get aggregate function maps
    map_claim = {
                    'Paid_Loss_Amount': np.sum,
                    'paid_Expenses_Amount': np.sum,
                    'Salvage_or_Subrogation?': np.sum,
                    'At_Fault?': np.mean,
                    'Deductible': np.sum,
                    'number_of_claimants': np.sum,
                    'Claim_Number': lambda x: x.nunique()
                }
    # select claims relevant to sample train data, aggregate at policy number level
    agg_claim = df_claim.groupby(level=[0, 1]).agg(map_claim)
    agg_claim = agg_claim.unstack(level=1)

    return(agg_claim)


def create_sample_data_main_coverage(df_train, df_policy, df_claim, df_map_coverage, data_size=2000):
    '''
    In:
        DataFrame(df_train), # contains dependent variable
        DataFrame(df_policy),
        DataFrame(df_claim),
        DataFrame(df_coverage), # maps coverage to main coverage group
        int(data_size), # sample size of df_train

    Out:
        DataFrame(df_sample),

    Description:
        Take a few samples and aggregate policy info and claim info
        Data should be ready for model
    '''
    # sample train data
    df_sample = df_train.sample(n=data_size, random_state=0)
    df_policy = df_policy.loc[df_sample.index]
    df_claim = df_claim.loc[df_sample.index]

    # aggregate policy info by Policy_Number
    agg_policy = get_main_coverage_aggregated_policy(df_policy)

    # aggregate claim info by Policy_Number
    agg_claim = get_main_coverage_aggregated_claim(df_claim, df_map_coverage)

    # merge train, policy, and claim data
    df_sample = df_sample.merge(agg_policy, how='left', left_index=True, right_index=True)
    df_sample = df_sample.merge(agg_claim, how='left', left_index=True, right_index=True)

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
    #df_policy = read_raw_data('policy_0702.csv')
    #df_map_coverage = read_raw_data('coverage_map.csv', index_col='Coverage')

    df_sample = create_sample_data_main_coverage(df_train, df_policy, df_claim, df_map_coverage, data_size=2000)
    write_sample_data(df_sample)