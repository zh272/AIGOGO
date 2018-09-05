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
def get_bs2_feature_selection(df_policy, df_claim):
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

    print('\nSplitting train valid label\n')
    y_train, y_valid = train_test_split(y_train_all, test_size=0.2, random_state=0)

    print('Getting neural network processed premiums')
    X_fs = read_data('premium_60_1.csv')

    # insured


    print('\nSplitting train valid test features\n')
    X_train = X_fs.loc[y_train.index]
    X_valid = X_fs.loc[y_valid.index]

    print('Writing results to file')
    write_data(X_train, "X_train_fs.csv")
    write_data(y_train, "y_train_fs.csv")
    write_data(X_valid, "X_valid_fs.csv")
    write_data(y_valid, "y_valid_fs.csv")

    return(None)

