import numpy as np
import os
import pandas as pd


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