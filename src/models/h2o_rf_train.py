import h2o
import os
import pandas as pd
from h2o.estimators.random_forest import H2ORandomForestEstimator

def train_h2orf_sample(h2o_sample):
    '''
    In:
        H2OFrame(h2o_sample)
    Out:
        Any(rf_v1) -> random forest model
    Description:
        basic random forest model on samples
    '''
    # split train, valid, and test data set
    train, valid, test = h2o_sample.split_frame([0.6, 0.2], seed=0)
    # separate independent variables from dependent variables
    col_y = 'Next_Premium'
    col_X = [col for col in h2o_sample.columns if col != col_y]
    # create random forest model
    rf_v1 = H2ORandomForestEstimator(
        model_id="rf_v1",
        ntrees=200,
        stopping_metric='mae',
        stopping_rounds=2,
        score_each_iteration=True,
        seed=1000000)
    # train random forest model
    rf_v1.train(col_X, col_y, training_frame=train, validation_frame=valid)

    # get model output
    output = {'model': rf_v1,
              'test_mae': (rf_v1.predict(test[col_X]) - test[col_y]).abs().mean(),
              }

    return(output)

def get_h2orf_submission(X_train, y_train, X_test, y_test):
    '''
    In:
        H2OFrame(h2o_sample)
    Out:
        Any(rf_v1) -> random forest model
    Description:
        basic random forest model for submission
    '''
    df_train = y_train.merge(X_train, how='left', left_index=True, right_index=True)
    h2o_train = h2o.H2OFrame(df_train)
    # split train into train and valid
    train, valid = h2o_train.split_frame(ratios = [0.8], seed=0)
    # separate independent variables from dependent variables
    col_y = 'Next_Premium'
    col_X = list(X_train.columns)
    # create random forest model
    rf_v1 = H2ORandomForestEstimator(
        model_id="rf_v1",
        ntrees=200,
        stopping_metric='mae',
        stopping_rounds=2,
        score_each_iteration=True,
        seed=1000000)
    # train random forest model
    rf_v1.train(col_X, col_y, training_frame=train, validation_frame=valid)

    df_test = y_test.merge(X_train, how='left', left_index=True, right_index=True)
    test = h2o.H2OFrame(df_test, column_types=h2o_train.types)
    submission = rf_v1.predict(test[col_X]).as_data_frame()
    submission['Policy_Number'] = y_test.index
    submission = submission.set_index(['Policy_Number'])
    submission.columns = ['Next_Premium']

    return({'model': rf_v1, 'submission': submssion})


def read_interim_data(file_name, index_col='Policy_Number'):
    '''
    In: file_name

    Out: interim_data

    Description: read data from directory /data/interim
    '''
    # set the path of raw data
    interim_data_path = os.path.join(os.path.pardir, os.path.pardir, 'data', 'interim')

    file_path = os.path.join(interim_data_path, file_name)
    interim_data = pd.read_csv(file_path, index_col=index_col)

    return(interim_data)

def write_precessed_data(df):
    '''
    In:
        DataFrame(df),
        str(file_name),
    Out:
        None

    Description:
        Write sample data to directory /data/interim
    '''
    precessed_data_path = os.path.join(os.path.pardir, os.path.pardir, 'data', 'precessed')
    write_sample_path = os.path.join(precessed_data_path, 'testing-set.csv')
    df.to_csv(write_sample_path)

    return(None)

if __name__ == '__main__':

    # ### Start H2O
    # Start up a 1-node H2O cloud on your local machine, and allow it to use all CPU cores and up to 2GB of memory:

    h2o.init(max_mem_size = "2G")             #specify max number of bytes. uses all cores by default.
    h2o.remove_all()                          #clean slate, in case cluster was already running

    X_train_id = read_interim_data('X_train_id.csv')
    X_test_id = read_interim_data('X_test_id.csv')
    y_train_id = read_interim_data('y_train_id.csv')

    #h2o_sample = read_interim_data('sample-set-id.csv')
    #mod_train_sample = train_h2orf_sample(h2o_sample)

    mod_output = get_h2orf_submission(X_train_id, y_train_id, X_test_id)

    write_precessed_data(model_output['submission'])

    h2o.shutdown(prompt=False)