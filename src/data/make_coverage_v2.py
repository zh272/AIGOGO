import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# 16G, 16P
# Premium = exp(c + a1 * I(distr_dir) + a2 * I(qpt<=3) + a3 * I(legal)) * (1 + plia_acc) * Insured_Amount_adj(IAs)

# 29B, 29K
# Premium = exp(c + a1 * I(distr_dir) + a2 * I(qpt<=3) + a3 * I(legal)) * Insured_Amount_adj(IAs)

# 05N, 09@
# Premium = exp(c + a1 * I(distr_dir) + a2 * I4(year_label) + a3 * I32(thf_factor)) * IA3 * f(Coverage)

# 20K
# Premium = exp(c + a1 * I(distr_dir) + a2 * (I(legal))

# 18@
# Premium = exp(c + a1 * I(distr_dir)) * Insured_Amount_adj(IAs)

# 04M, 05E(partially explained)
# Premium = exp(c + a1 * I(distr_dir) + a2 * f(vehicle_code)) * (age_fac * sex_fac - pdmg_acc) * IA3 * f(Coverage)

# Rest Not explained

df_coverage = get_id_aggregated_coverage(df_policy)

def get_id_aggregated_coverage(df_policy):
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
    agg_premium = agg_premium.unstack(level=1)
    agg_premium.columns = ['cpremium_dmg_acc_adj', 'cpremium_lia_acc_adj', 'cpremium_acc_adj_thf']

    return(agg_premium[['cpremium_dmg_acc_adj', 'cpremium_lia_acc_adj']])