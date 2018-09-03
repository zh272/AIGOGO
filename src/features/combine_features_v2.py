from sklearn.model_selection import train_test_split
import sys
sys.path.insert(0, '../data')
from utils import read_data, write_data

######## get pre feature selection data set ########
def get_combined_features(df_policy, df_claim):
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

    print('\nSplitting train valid label\n')
    y_train, y_valid = train_test_split(y_train_all, test_size=0.2, random_state=0)

    print('Getting neural network processed premiums')
    X_fs = read_data('premium_60_1.csv')

    print('\nSplitting train valid test features\n')
    X_train_all = X_fs.loc[y_train_all.index]
    X_train = X_fs.loc[y_train.index]
    X_valid = X_fs.loc[y_valid.index]
    X_test = X_fs.loc[y_test.index]

    print('Writing results to file')
    write_data(X_train, "X_train_bs2.csv")
    write_data(y_train, "y_train_bs2.csv")
    write_data(X_valid, "X_valid_bs2.csv")
    write_data(y_valid, "y_valid_bs2.csv")
    write_data(X_train_all, "X_train_all_bs2.csv")
    write_data(y_train_all, "y_train_all_bs2.csv")
    write_data(X_test, "X_test_bs2.csv")
    write_data(y_test, "y_test_bs2.csv")


    write_data()

    return(None)