import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

'''
1. Insured Amount aggregation
2. Distribution channel factor
3. plia_acc, pdmg_acc change factor
'''

col_cov = 'Insurance_Coverage'
col_IA1 = 'Insured_Amount1'
col_IA2 = 'Insured_Amount2'
col_IA3 = 'Insured_Amount3'
cols_cov = ['Insurance_Coverage', 'Insured_Amount1', 'Insured_Amount2', 'Insured_Amount3', 'Premium', 'Coverage_Deductible_if_applied']

coverages = df_policy[col_cov].value_counts()
df_policy_adj = get_claims_per_policy(df_policy, df_claim, '16G')
df_policy_adj = get_adjusted_lia_dmg_acc(df_policy_adj)

# get claims per insurer/vehicle
def get_claims_per_policy(df_policy, df_claim):
    '''
    In:
        DataFrame(df_policy),
        DataFrame(df_claim),

    Out:
        DataFrame(df_policy),

    Description:
        get number of claims received by insured or vehicle
    '''
    # get claim count by vehicle
    df_claim = df_claim.groupby(['Vehicle_identifier']).agg({'Claim_Number': lambda x: x.nunique()})
    df_claim.columns = ['cclaim_num_adj']
    # get insured's ID and assured by vehicle
    df_insured = df_policy.groupby(['Vehicle_identifier']).agg({"Insured's_ID": lambda x: x.iloc[0], 'fassured': lambda x: x.iloc[0]})
    # append claim number to insured for natural or vehicle for legal
    df_insured_claim = df_insured.merge(df_claim, how='left', left_index=True, right_index=True)
    df_insured_leg = df_insured_claim[df_insured_claim['fassured'] % 2 == 0][['cclaim_num_adj']]

    df_insured_nat = df_insured_claim[df_insured_claim['fassured'] % 2 == 1]
    df_insured_nat = df_insured_nat.groupby(["Insured's_ID"]).agg({'cclaim_num_adj': lambda x: len(x)})
    # append claim number to policy
    df_policy_leg = df_policy[df_policy['fassured'] % 2 == 0]
    df_policy_nat = df_policy[df_policy['fassured'] % 2 == 1]

    df_policy_leg = df_policy_leg.merge(df_insured_leg, how='left', left_on='Vehicle_identifier', right_index=True)
    df_policy_nat = df_policy_nat.merge(df_insured_nat, how='left', left_on="Insured's_ID", right_index=True)
    # concat policy of natural and legal
    df_policy = pd.concat([df_policy_leg, df_policy_nat])
    df_policy['cclaim_num_adj'] = df_policy['cclaim_num_adj'].where(~np.isnan(df_policy['cclaim_num_adj']), 0)

    return(df_policy)

def get_adjusted_lia_dmg_acc(df_policy):
    '''
    In:
        DataFrame(df_policy), # includes claim information

    Out:
        DataFrame(df_policy),

    Description:
        get plia_acc_adj and pdmg_acc_adj adjusted to the number of claims
    '''
    # initialize liability and damage claim multiplier
    lia_acc_list = [-0.38, -0.35, -0.3, -0.2, -0.1, 0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.8, 3.2, 3.7, 4.2, 4.7]
    dmg_acc_list = [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.6, 2, 4]
    #lia_class_list = list(range(-1, 20))

    def get_adj(x, lst, col, ssize):
        # every claim increase 3 steps, if no claims, decrease one step
        steps = ssize * x['cclaim_num_adj']
        steps = np.where(steps == 0, -1, steps)
        # find original place, add steps to it
        idx = x[col].map(lambda y: lst.index(y)) + steps
        idx = np.where(idx < 0, 0, idx)
        idx = np.where(idx >= len(lst), len(lst) - 1, idx)

        adj = [lst[int(i)] for i in idx ]
        return(adj)


    df_policy = df_policy.assign(plia_acc_adj = lambda x: get_adj(x, lia_acc_list, 'plia_acc', 3))
    df_policy = df_policy.assign(pdmg_acc_adj = lambda x: get_adj(x, dmg_acc_list, 'pdmg_acc', 1))
    #df_policy = df_policy.assign(lia_class_adj = lambda x: get_lia_class_adj(x))

    return(df_policy)


def get_distr_multi_and_scaled_insured(df_cov):
    '''
    In:
        DataFrame(df_cov), # df_policy by coverage, with additional 'Premium_dir' and 'Premium_indir'

    Out:
        DataFrame(df_cov),

    Description:
        get policy, coverage level distribution multiplier (1/0.65 or 1/0.7927) and scaled insured amount
    '''
    bucket_size = 20
    # find all insured amounts combinations
    cols_ia = ['Insured_Amount1','Insured_Amount2','Insured_Amount3', 'Coverage_Deductible_if_applied']
    insured_amounts = df_cov[cols_ia].drop_duplicates(cols_ia)
    # loop through all insured amounts combinations
    sample_arr = []
    for ias in insured_amounts.itertuples():
        sample = df_cov[(df_cov['Insured_Amount1'] == ias[1]) & (df_cov['Insured_Amount2'] == ias[2]) & (df_cov['Insured_Amount3'] == ias[3]) & (df_cov['Coverage_Deductible_if_applied'] == ias[4])]
        # bucket premiums
        Premium_bdir = round(sample['Premium_dir'] / bucket_size) * bucket_size
        Premium_bindir = round(sample['Premium_indir'] / bucket_size) * bucket_size
        map_freq = pd.concat([Premium_bdir, Premium_bindir]).value_counts()
        # get distribution method: False for indirect; True for direct
        Distr_Direct = Premium_bdir.map(map_freq) >= Premium_bindir.map(map_freq)
        sample['cdistr_multi'] = np.where(Distr_Direct, 1/0.7925, 1/0.65)
        sample['cinsured_scaled'] = np.where(Distr_Direct, sample['Premium_dir'], sample['Premium_indir'])
        sample_arr.append(sample)
    df_cov = pd.concat(sample_arr)
    # get scaled insured amount
    df_cov['cinsured_scaled'] = df_cov['cinsured_scaled'] / df_cov['cinsured_scaled'].max()

    return(df_cov)


def get_id_coverage_16G(df_policy_adj, coverage):
    '''
    In:
        DataFrame(df_policy_ad), # df_policy with 'plia_acc_adj' & 'pdmg_acc_adj'

    Out:
        DataFrame(df),

    Description:
        calculate scaled insured amount, distribution multiplier, and adjusted premium
        this calculation apply to 16G, 16P
        Premium = exp(c + I(distr_dir)) * (1 + plia_acc) * Insured_Amount_adj(qpt, fassured, IAs)
    '''
    cols_cov = ['cinsured_scaled', 'cdistr_multi', 'Premium_adj']
    df_raw = df_policy_adj[df_policy_adj['Insurance_Coverage'] == coverage]

    # get adj premium
    df['Premium_adj'] = df['Premium']

    # get scaled instured amount and distribution multiplier
    df = df.assign(Premium_wo_acc = lambda x: x['Premium'] / (1 + x['plia_acc']))
    df = df.assign(Premium_dir = lambda x: x['Premium_wo_acc'] * 0.7927)
    df = df.assign(Premium_indir = lambda x: x['Premium_wo_acc'] * 0.65)
    df = get_distr_multi_and_scaled_insured(df)

    # bind created variables with raw
    df = df[cols_cov]
    df = df_raw.merge(df, how='left', left_index=True, right_index=True)

    return(df)


def get_id_coverage_29K(df_policy_adj, coverage):
    '''
    In:
        DataFrame(df_policy_ad), # df_policy with 'plia_acc_adj' & 'pdmg_acc_adj'

    Out:
        DataFrame(df),

    Description:
        calculate scaled insured amount, distribution multiplier, and adjusted premium
        this calculation apply to 29K， 29B
        Premium = exp(c + I(distr_dir)) * Insured_Amount_adj(qpt, fassured, IAs)
    '''
    cols_cov = ['cinsured_scaled', 'cdistr_multi', 'Premium_adj']
    df_raw = df_policy_adj[df_policy_adj['Insurance_Coverage'] == coverage]

    # get adj premium
    df = df_raw.assign(Premium_adj = lambda x: x['Premium'])

    # get scaled instured amount and distribution multiplier
    df = df.assign(Premium_dir = lambda x: x['Premium'] * 0.7927)
    df = df.assign(Premium_indir = lambda x: x['Premium'] * 0.65)
    df = get_distr_multi_and_scaled_insured(df)

    # bind created variables with raw
    df = df[cols_cov]
    df = df_raw.merge(df, how='left', left_index=True, right_index=True)

    return(df)


def get_id_coverage_5N(df_policy_adj, coverage):
    '''
    In:
        DataFrame(df_policy_ad), # df_policy with 'plia_acc_adj' & 'pdmg_acc_adj'

    Out:
        DataFrame(df),

    Description:
        calculate scaled insured amount, distribution multiplier, and adjusted premium
        this calculation apply to 05N
        Premium = exp(c + I(distr_dir) + f(Vehicle_Code) + f(year_lab)) * Insured_Amount_adj(IAs)
    '''
    cols_cov = ['cinsured_scaled', 'cdistr_multi', 'Premium_adj']
    df_raw = df_policy_adj[df_policy_adj['Insurance_Coverage'] == coverage]

    # get year label, and 05N multiplier
    df = df_raw
    df['year_lab'] = np.where(df['Manafactured_Year_and_Month']<= 2011, 3, 2014 - df['Manafactured_Year_and_Month'])
    df['05N_Multi'] = np.where(df['Insured_Amount3'] != 0, df['Premium'] * 10000 / df['Insured_Amount3'], 0)

    # get year label parameter
    df_freq = df['Coding_of_Vehicle_Branding_&_Type'].value_counts()
    df_year = df[df['Coding_of_Vehicle_Branding_&_Type']==df_freq.index[0]]
    df_year = df_year[df_year['Insured_Amount3'] != 0]
    df_year = df_year.groupby(['year_lab']).agg({'05N_Multi': np.median})

    # manual adjustment
    df_year.columns = ['year_multi']
    df_year = df_year['year_multi']
    df_year[0] = df_year[0] * .7925 / .65
    for i in range(0, 4):
        df_year[i] /= df_year[min(i+1, 3)]

    df_year.columns = ['year_multi']
    df_year = df_year['year_multi'] / df_year['year_multi'][3]


    # get premium adj

    X_train_id.columns
    df_view = X_train_id[X_train_id['cbucket'] == '16G']
    df_view = df_view.merge(y_train_id, how='left', left_index=True, right_index=True)
    df_view = df_view.merge(df_16G[['plia_acc', 'Premium_adj', 'plia_acc_adj']], how='left', left_index=True, right_index=True)
    df_view = df_view[['Next_Premium', 'Premium_adj', 'cpremium_lia', 'plia_acc_adj', 'plia_acc']]

    df_view['acc_implied'] = df_view['Next_Premium'] / df_view['cpremium_lia'] * (df_view['plia_acc'] + 1) - 1


    df_view.to_csv(write_sample_path)
    # add year_multi back to df
    year_multi = df['year_lab'].map(df_year)
    df['Multi_year_adj'] = df['05N_Multi'] / df['year_lab'].map(df_year)

    # get adj premium
    df = df_raw.assign(Premium_adj = lambda x: x['Premium'])

    # get scaled instured amount and distribution multiplier
    df = df.assign(Premium_dir = lambda x: x['Premium'] * 0.7927)
    df = df.assign(Premium_indir = lambda x: x['Premium'] * 0.65)
    df = get_distr_multi_and_scaled_insured(df)

    # bind created variables with raw
    df = df[cols_cov]
    df = df_raw.merge(df, how='left', left_index=True, right_index=True)

    return(df)

df_view = df_policy[(df_policy['Insurance_Coverage']=='05N') & (df_policy['Coding_of_Vehicle_Branding_&_Type']=='ee9bc138fac33cb145e948cb6f6324975a61136f')]

df_view = df_policy[(df_policy['Insurance_Coverage']=='05N')]
df_view = df_view[df_view['Manafactured_Year_and_Month'] == 2012]
df_view = df_view[df_view['Insured_Amount3'] != 0]

plt.hist(df_view['Premium'] * 1000000 / df_view['Insured_Amount3'], bins = 100)

df_view = df_policy['Insurance_Coverage'].value_counts()
df_view = df_policy[df_policy['Insurance_Coverage']=='05N']['Coding_of_Vehicle_Branding_&_Type'].value_counts()
df_16G = get_id_coverage_16G(df_policy_adj, '16G')

df_view = df_16G.merge(y_train_id, left_index=True, right_index=True)
sum(df_view['Next_Premium'] == df_view['Premium_adj'])
df_view = X_train_id['16G']

df_16P = get_id_coverage_16G(df_policy_adj, '16P')
df_policy[(df_policy['Insurance_Coverage']=='05N')&(df_policy['Coding_of_Vehicle_Branding_&_Type']==df_view.index[0])].head(2000).to_csv(write_sample_path)
write_sample_by_coverage('55J', top=True)
def write_sample_by_coverage(coverage, top=False):
    global df_policy

    if top:
        file_name = 'sample_{}_top.csv'.format(coverage)
        df = df_policy[(df_policy['Insurance_Coverage']==coverage) & (df_policy['Coding_of_Vehicle_Branding_&_Type']=='ee9bc138fac33cb145e948cb6f6324975a61136f')]
    else:
        file_name = 'sample_{}.csv'.format(coverage)
        df = df_policy[(df_policy['Insurance_Coverage']==coverage)].head(2000)
    #& (df_policy['Coding_of_Vehicle_Branding_&_Type']=='ee9bc138fac33cb145e948cb6f6324975a61136f')]

    interim_data_path = os.path.join(os.path.pardir, os.path.pardir, 'data', 'interim')
    write_sample_path = os.path.join(interim_data_path, file_name)
    df.to_csv(write_sample_path)

    return(None)

