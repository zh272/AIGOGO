from create_features_v2 import *
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import fire
import lightgbm as lgb
import pandas as pd
import sys
sys.path.insert(0, '../data')
from utils import read_data, write_data

'''
base case: 1799

+ premium by main coverage grp: 1799
+ premium on 0 insured amounts: 1799
+ premium avg of insured coverage: 1800
+ premium avg of insured coverage by vmy: 1799

+ real_acc_dmg: 1787
+ real_acc_lia: 1795
+ real_acc_dmg & lia: 1784

+ real_age: 1793
+ real_age_grp: 1794
+ real_age_tail: 1793

+ real_mc_mean_sex: 1790
+ real_mc_prob_sex: 1789

+ real_mc_mean_marriage: 1798
+ real_mc_prob_marriage: 1799

+ real_mc_mean_sex_marriage: 1791

+ real_mc_prob_sex & mean_marriage: 1791
+ real_mc_prob_sex & prob_marriage: 1791

+ real_mc_sex + real_age: 1787

+ real_mc_mean_zip: 1801
+ real_mc_prob_zip: 1803

+ replacement cost: 1795
'''

######## get pre feature selection data set ########
def get_bs2_combined_features(df_policy, df_claim):
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
    print('Getting labels')
    y_train_all = read_data('training-set.csv', path='raw')
    y_test = read_data('testing-set.csv', path='raw')

    print('Getting neural network processed premiums')
    X_fs = read_data('premium_60_1.csv')

    print('\nSplitting train valid test features\n')
    X_train_all = X_fs.loc[y_train_all.index]
    X_test = X_fs.loc[y_test.index]

    print('Writing results to file')
    write_data(X_train_all, "X_train_all_bs2.csv")
    write_data(y_train_all, "y_train_all_bs2.csv")
    write_data(X_test, "X_test_bs2.csv")
    write_data(y_test, "y_test_bs2.csv")

    return(None)


def demo():
    df_claim = read_data('claim_0702.csv', path='raw')
    df_policy = read_data('policy_0702.csv', path='raw')

    get_bs2_combined_features(df_policy, df_claim)

if __name__ == '__main__':
    fire.Fire(demo)