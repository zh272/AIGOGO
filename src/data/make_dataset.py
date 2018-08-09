import numpy as np
import os
import pandas as pd


######## aggregation and helper func ########
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
    agg_coverage = agg_coverage.unstack(level=1, fill_value=0)

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
    agg_claim = agg_claim.unstack(level=1, fill_value=0)

    return(agg_claim)


def get_id_aggregated_insured(df_policy, df_claim):
    '''
    In:
        DataFrame(df_policy),
        DataFrame(df_claim),

    Out:
        DataFrame(agg_insured),

    Description:
        (1) transform birth dates into age groups
        (2) aggregate claim data at insured level: count, paid amount, salvage amount
        (3) aggregate policy data at insured level: count, average coverage of each policy, average premium of each policy
        (4) aggregate vehicle data at insured level: average replacement cost of each vehicle
    '''
    # transform basic info
    def get_age_label_id(ibirth):
        if pd.isnull(ibirth):
            return np.nan
        else:
            age = 2015 - int(ibirth[3:]);
            if age < 25:
                return 1.74
            elif age < 30:
                return 1.15
            elif age < 60:
                return 1
            elif age < 70:
                return 1.07
            else:
                return 1.07

    agg_basic = df_policy[["Insured's_ID", 'fsex', 'fmarriage']]
    agg_basic['isex_lab'] = df_policy['fsex'].map(lambda x: 0.9 if x==2 else 1)
    agg_basic['iage_lab'] = df_policy['ibirth'].map(get_age_label_id)
    agg_basic['iassured_lab'] = df_policy['fassured'].map(lambda x: x % 2)
    map_agg_basic = {'fmarriage': lambda x: x.iloc[0],
                     'isex_lab': lambda x: x.iloc[0],
                     'iage_lab': lambda x: x.iloc[0],
                     'iassured_lab': lambda x: x.iloc[0],
                     }
    agg_basic = agg_basic.groupby(by=["Insured's_ID"]).agg(map_agg_basic)
    agg_insured = agg_basic

    # transform policy and vehicle info
    agg_policy = df_policy[["Insured's_ID", 'Insurance_Coverage', 'Premium', 'Replacement_cost_of_insured_vehicle']]
    # 1. aggregate at policy level
    map_agg_policy = {"Insured's_ID": lambda x: x.iloc[0],
                      'Insurance_Coverage': lambda x: len(x),
                      'Premium': np.sum,
                      'Replacement_cost_of_insured_vehicle': lambda x: x.iloc[0]
                      }
    agg_policy = agg_policy.groupby(level=0).agg(map_agg_policy)
    agg_policy["Policy's_ID"] = agg_policy.index
    # 2. average at insured's level
    map_agg_policy = {'Insurance_Coverage': np.mean,
                      'Premium': np.mean,
                      'Replacement_cost_of_insured_vehicle': np.mean,
                      "Policy's_ID": lambda x: len(x),
                      }
    agg_policy = agg_policy.groupby(["Insured's_ID"]).agg(map_agg_policy)
    agg_policy.columns = ['ipolicy_coverage_avg', 'ipolicy_premium_avg', 'ivehicle_repcost_avg', 'ipolicies']
    agg_insured = agg_insured.merge(agg_policy, how='left', left_index=True, right_index=True)

    # transform claim info
    agg_claim = df_claim[['Claim_Number', 'Paid_Loss_Amount', 'paid_Expenses_Amount', 'Salvage_or_Subrogation?']]
    #1. map insured's id to claim
    agg_claim = agg_claim.merge(df_policy[["Insured's_ID"]], how='left', left_index=True, right_index=True)
    #2. sum up paid loss amount and paid expense amount
    agg_claim['Paid_Amount'] = agg_claim['Paid_Loss_Amount'] + agg_claim['paid_Expenses_Amount']
    #3. aggregate at policy level
    map_agg_claim = {'Claim_Number': lambda x: x.nunique(),
                     'Paid_Amount': np.sum,
                     'Salvage_or_Subrogation?': np.sum}
    agg_claim = agg_claim.groupby(by=["Insured's_ID"]).agg(map_agg_claim)
    agg_claim.columns = ['iclaims', 'iclaim_paid_amount', 'iclaim_salvage_amount']
    agg_insured = agg_insured.merge(agg_claim, how='left', left_index=True, right_index=True)

    return(agg_insured)


def get_id_aggregated_vehicle(df_policy):
    '''
    In:
        DataFrame(df_policy),

    Out:
        DataFrame(agg_vehicle),

    Description:
        (1) placeholder for further info extraction
        (2) tranform manufacture year to vehicle age group
        (3) separate domestic vehicle from imported
        (4) separate locomotive vehicle from others
        (5) group engine displacement
    '''
    agg_vehicle = df_policy[['Vehicle_identifier', 'Coding_of_Vehicle_Branding_&_Type', 'Vehicle_Make_and_Model1', 'Vehicle_Make_and_Model2']]

    def get_vehicleage_label_id(vyear):
        vyear = 2015 - vyear
        if vyear <= 3:
            return vyear
        else:
            return round((vyear - 3) / 5) + 4

    agg_vehicle['vlocomotive'] = df_policy['qpt'].map(lambda x: x<4)
    agg_vehicle['vyear_lab'] = df_policy['Manafactured_Year_and_Month'].map(get_vehicleage_label_id)
    agg_vehicle['vregion_lab'] = df_policy['Imported_or_Domestic_Car'].map(lambda x: x==10)
    agg_vehicle['vengine_lab'] = df_policy['Engine_Displacement_(Cubic_Centimeter)'].map(lambda x: int(x/1000))
    agg_vehicle = agg_vehicle.groupby(level=0).agg(dict.fromkeys(agg_vehicle.columns, lambda x: x.iloc[0]))

    return(agg_vehicle)


def get_id_aggregated_main_coverage(df_policy):
    '''
    In:
        DataFrame(df_policy),

    Out:
        DataFrame(agg_coverage),

    Description:
        (1) placeholder for further info extraction
        (2) tranform manufacture year to vehicle age group
        (3) separate domestic vehicle from imported
        (4) separate locomotive vehicle from others
    '''
    # get bucket coverage ID at policy level
    agg_coverage = df_policy.groupby(level=0).agg({'Insurance_Coverage': lambda x: '|'.join(list(x[~x.sort_values().isnull()].unique()))})
    agg_coverage.columns = ['cbucket']

    # get premium by main coverage group
    map_agg_premium = {'車損': 'Dmg',
                       '竊盜': 'Thf',
                       '車責': 'Lia'}
    # 1. group premium by Main_Insurance_Coverage_Group
    agg_premium = df_policy[['Main_Insurance_Coverage_Group', 'Premium']]
    agg_premium['Main_Insurance_Coverage_Group'] = agg_premium['Main_Insurance_Coverage_Group'].map(map_agg_premium)
    agg_premium = agg_premium.set_index(['Main_Insurance_Coverage_Group'], append = True)
    agg_premium = agg_premium.groupby(level=[0,1]).agg({'Premium': np.sum})
    # 2. aggregate at policy level
    agg_premium = agg_premium.unstack(level=1, fill_value=0)
    agg_premium.columns = ['cpremium_dmg', 'cpremium_lia', 'cpremium_thf']

    agg_coverage = agg_coverage.merge(agg_premium, how='left', left_index=True, right_index=True)

    return(agg_coverage)


def get_id_aggregated_coverage_v1(df_policy):
    '''
    In:
        DataFrame(df_policy),

    Out:
        DataFrame(agg_coverage),

    Description:
        get acc adjusted premium
    '''
    coverage_all = list(df_policy['Insurance_Coverage'].value_counts().index)

    # group 1:
    coverage_g1 = ['16G', '16P']
    df_g1 = df_policy[df_policy['Insurance_Coverage'].isin(coverage_g1)]
    df_g1['premium_acc_adj'] = df_g1['Premium'] / (1 + df_g1['plia_acc'])

    def get_age_label_id(ibirth):
        if pd.isnull(ibirth):
            return 1
        else:
            age = 2015 - int(ibirth[3:]);
            if age < 25:
                return 1.74
            elif age < 30:
                return 1.15
            elif age < 60:
                return 1
            elif age < 70:
                return 1.07
            else:
                return 1.07

    # group 2:
    coverage_g2 = ['04M', '05E', '55J']
    df_g2 = df_policy[df_policy['Insurance_Coverage'].isin(coverage_g2)]
    g2_age_fac = df_g2['ibirth'].map(get_age_label_id)
    g2_sex_fac = df_g2['fsex'].map(lambda x: 0.9 if x == '2' else 1)
    df_g2['premium_acc_adj'] = df_g2['Premium'] / (g2_age_fac * g2_sex_fac + df_g2['pdmg_acc'])

     # group 3:
    #coverage_g3 = ['29B', '29K', '5N', '20K', '18@', '09@', '12L', '15F']
    coverage_g3 = [cov for cov in coverage_all if ((cov not in coverage_g1) & (cov not in coverage_g2))]
    df_g3 = df_policy[df_policy['Insurance_Coverage'].isin(coverage_g3)]
    df_g3['premium_acc_adj'] = df_g3['Premium']

    df_coverage = pd.concat([df_g1, df_g2, df_g3])

    # aggregate coverage
    map_agg_premium = {'車損': 'Dmg',
                       '竊盜': 'Thf',
                       '車責': 'Lia'}
    # 1. group premium by Main_Insurance_Coverage_Group
    agg_premium = df_coverage[['Main_Insurance_Coverage_Group', 'premium_acc_adj']]
    agg_premium['Main_Insurance_Coverage_Group'] = agg_premium['Main_Insurance_Coverage_Group'].map(map_agg_premium)
    agg_premium = agg_premium.set_index(['Main_Insurance_Coverage_Group'], append = True)
    agg_premium = agg_premium.groupby(level=[0,1]).agg({'premium_acc_adj': np.sum})
    # 2. aggregate at policy level
    agg_premium = agg_premium.unstack(level=1, fill_value=0)
    agg_premium.columns = ['cpremium_dmg_acc_adj', 'cpremium_lia_acc_adj', 'cpremium_acc_adj_thf']

    return(agg_premium[['cpremium_dmg_acc_adj', 'cpremium_lia_acc_adj']])


def get_id_aggregated_coverage(df_policy):
    '''
    In:
        DataFrame(df_policy),

    Out:
        DataFrame(agg_coverage),

    Description:
        get premium by coverage
    '''
    agg_coverage = df_policy[['Insurance_Coverage', 'Premium']]
    agg_coverage = agg_coverage.set_index(['Insurance_Coverage'], append=True)
    agg_coverage = agg_coverage.unstack(level=1, fill_value=0)
    agg_coverage.columns = ['_'.join(col) for col in agg_coverage.columns]

    cols_imp = ['Premium_05E', 'Premium_55J', 'Premium_05N', 'Premium_04M', 'Premium_00I', 'Premium_29B']

    return(agg_coverage[cols_imp])


def get_id_aggregated_policy(df_policy):
    '''
    In:
        DataFrame(df_policy),

    Out:
        DataFrame(agg_policy),

    Description:
        get aggregated policy using the first row
    '''
    cols_policy = ['aassured_zip', 'iply_area', 'Distribution_Channel', 'Multiple_Products_with_TmNewa_(Yes_or_No?)']
    agg_policy = df_policy.groupby(level=0).agg(dict.fromkeys(cols_policy, lambda x: x.iloc[0]))

    return(agg_policy)


def get_id_aggregated_claim(df_claim):
    '''
    In:
        DataFrame(df_claim),

    Out:
        DataFrame(agg_claim),

    Description:
        get aggregated claim based on sums
    '''
    df_claim['closs'] = df_claim['paid_Expenses_Amount'] + df_claim['Paid_Loss_Amount']
    map_agg_claim = {'Vehicle_identifier': lambda x: x.iloc[0],
                     'Claim_Number': lambda x: x.nunique(),
                     'closs': np.sum,
                     'Salvage_or_Subrogation?': np.sum,
                     'Cause_of_Loss': lambda x: '|'.join(list(x[~x.sort_values().isnull()].unique()))}
    agg_claim = df_claim.groupby(level=0).agg(map_agg_claim)
    agg_claim.columns = ['cclaim_id', 'cclaims', 'closs', 'csalvate', 'ccause_type']

    return(agg_claim)


######## create data set func ########
def create_train_test_data_main_coverage(df_train, df_test, df_policy, df_claim, df_map_coverage):
    '''
    In:
        DataFrame(df_train), # contains dependent variable
        DataFrame(df_test),
        DataFrame(df_policy),
        DataFrame(df_claim),
        DataFrame(df_map_coverage), # maps coverage to main coverage group

    Out:
        DataFrame(df_sample),

    Description:
        create X, and y variables for training and testing
    '''
    # create train independent data
    train_policy = get_main_coverage_aggregated_policy(df_policy.loc[df_train.index])
    train_claim = get_main_coverage_aggregated_claim(df_claim.loc[df_train.index], df_map_coverage)
    X_train = train_policy.merge(train_claim, how='left', left_index=True, right_index=True)
    y_train = df_train

    # create test independent data
    test_policy = get_main_coverage_aggregated_policy(df_policy.loc[df_test.index])
    test_claim = get_main_coverage_aggregated_claim(df_claim.loc[df_test.index], df_map_coverage)
    X_test = test_policy.merge(test_claim, how='left', left_index=True, right_index=True)

    return(X_train, X_test, y_train)


def get_id_merged_data(df_policy, df_claim):
    '''
    In:
        DataFrame(df_policy),
        DataFrame(df_claim),

    Out:
        DataFramed(df_X),

    Description:
        merge independent variables
    '''
    cols_id = ["Insured's_ID"]
    df_X = df_policy.groupby(level=0).agg(dict.fromkeys(cols_id, lambda x: x.iloc[0]))
    df_policy = df_policy.loc[df_X.index]
    df_claim = df_claim.loc[df_X.index]

    # aggregate insurer info
    agg_insured = get_id_aggregated_insured(df_policy, df_claim)

    # aggregate vehicle info
    agg_vehicle = get_id_aggregated_vehicle(df_policy)

    # aggregate coverage info
    agg_main = get_id_aggregated_main_coverage(df_policy)

    # aggregate coverage info
    agg_coverage = get_id_aggregated_coverage(df_policy)

    # aggregate policy info
    agg_policy = get_id_aggregated_policy(df_policy)

    # aggregate claim info
    agg_claim = get_id_aggregated_claim(df_claim)

    # merge train, policy, and claim data
    df_X = df_X.merge(agg_insured, how='left', left_on="Insured's_ID", right_index=True)
    df_X = df_X.merge(agg_vehicle, how='left', left_index=True, right_index=True)
    df_X = df_X.merge(agg_main, how='left', left_index=True, right_index=True)
    df_X = df_X.merge(agg_coverage, how='left', left_index=True, right_index=True)
    df_X = df_X.merge(agg_policy, how='left', left_index=True, right_index=True)
    df_X = df_X.merge(agg_claim, how='left', left_index=True, right_index=True)

    return(df_X)


def create_train_test_data_id(df_train, df_test, df_policy, df_claim):
    '''
    In:
        DataFrame(df_train), # contains dependent variable
        DataFrame(df_policy),
        DataFrame(df_claim),

    Out:
        DataFrame(df_sample),

    Description:
        create X, and y variables for training and testing
    '''
    X_all = get_id_merged_data(df_policy, df_claim)

    # get train data
    X_train = X_all.loc[df_train.index]
    y_train = df_train

    # get test data
    X_test = X_all.loc[df_test.index]
    return(X_train, X_test, y_train)


######## create sample func ########
def create_sample_data_id(df_train, df_policy, df_claim, data_size=20000):
    '''
    In:
        DataFrame(df_train), # contains dependent variable
        DataFrame(df_policy),
        DataFrame(df_claim),
        int(data_size), # sample size of df_train

    Out:
        DataFrame(df_sample),

    Description:
        Take a few samples and aggregate policy info and claim info
        Data should be ready for model
    '''
    # sample train and test data
    np.random.seed(0)
    df_sample = df_train.sample(n=data_size, random_state=0)
    msk = np.random.rand(len(df_sample)) < 0.8
    df_train = df_sample[msk]
    df_test = df_sample[~msk]

    #df_sample = df_train
    X_train, X_test, y_train = create_train_test_data_id(df_train, df_test, df_policy, df_claim)
    y_test = df_test[X_test.index]

    return(X_train, X_test, y_train, y_test)


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
    #df_map_coverage = read_raw_data('coverage_map.csv', index_col='Coverage')

    #X_train, X_test, y_train = create_train_test_data_main_coverage(df_train, df_test, df_policy, df_claim, df_map_coverage)

    X_train, X_test, y_train = create_train_test_data_id(df_train, df_test, df_policy, df_claim)

    write_test_data(X_train, "X_train_id.csv")
    write_test_data(X_test, "X_test_id.csv")
    write_test_data(y_train, "y_train_id.csv")

