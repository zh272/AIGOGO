import h2o
import numpy as np
import os
import pandas as pd
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.xgboost import H2OXGBoostEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator

schema = {'cat_age': 'enum',
 'cat_area': 'enum',
 'cat_assured': 'enum',
 'cat_cancel': 'enum',
 #'cat_distr': 'enum',
 'cat_marriage': 'enum',
 'cat_sex': 'enum',
 'cat_vc': 'enum',
 'cat_vmm1': 'enum',
 'cat_vmm2': 'enum',
 'cat_vmy': 'enum',
 'cat_vqpt': 'enum',
 'cat_vregion': 'enum',
 'cat_zip': 'enum',
 'int_acc_lia': 'int',
 'int_claim_plc': 'int',
 'int_others': 'int',
 'real_acc_dmg': 'real',
 'real_acc_lia': 'real',
 'real_loss_plc': 'real',
 'real_prem_dmg': 'real',
 'real_prem_ins': 'real',
 'real_prem_lia': 'real',
 'real_prem_plc': 'real',
 'real_prem_thf': 'real',
 'real_prem_vc': 'real',
 'real_vcost': 'real',
 'real_ved': 'real'}

######## get model input func ########
def get_train_input(train_only=False, ext='bs', seed=0):
    '''
    In:
        bool(train_only),
        int(seed)

    Out:
        DataFrame(X_train),
        DataFrame(X_test),
        DataFrame(y_train),
        DataFrame(y_test),

    Description:
        if train_only, then split train data into 80/20
        else read in train and test data
    '''
    if train_only:
        np.random.seed(seed)
        X_all = read_interim_data('X_train_{}.csv'.format(ext))
        y_all = read_interim_data('y_train_{}.csv'.format(ext))

        msk = np.random.rand(len(X_all)) < 0.8
        X_train = X_all[msk]
        y_train = y_all[msk]
        X_test = X_all[~msk]
        y_test = y_all[~msk]
    else:
        X_train = read_interim_data('X_train_{}.csv'.format(ext))
        X_test = read_interim_data('X_test_{}.csv'.format(ext))
        y_train = read_interim_data('y_train_{}.csv'.format(ext))
        y_test = read_raw_data('testing-set.csv')

    return(X_train, X_test, y_train, y_test)


######## train model func ########
def train_h2o_model(X_train, X_test, y_train, model, params):
    '''
    In:
        DataFrame(X_train),
        DataFrame(X_test),
        DataFrame(y_train)
    Out:
        dict(output) -> includes model, fit_test, fit_train
    Description:
        train h2o random forest model
    '''
    global schema
    # transform to h2o format
    df_train = y_train.merge(X_train, how='left', left_index=True, right_index=True)
    h2o_train = h2o.H2OFrame(df_train, column_types=schema)

    # split train into train and valid
    train, valid = h2o_train.split_frame(ratios = [0.8], seed=0)
    # separate independent variables from dependent variables
    col_y = 'Next_Premium'
    col_X = list(X_train.columns)
    # create random forest model
    rf_v1 = cv_h2o(col_X, col_y, train, valid, model, params)

    # fit model to train and test data
    output = {'model': rf_v1,
              'fit_train': get_fit_data(rf_v1, X_train, schema),
              'fit_test': get_fit_data(rf_v1, X_test, schema)
              }

    return(output)


######## get cross validation func ########
def cv_h2o(col_X, col_y, train, valid, model, params):
    '''
    In:
        list(col_X),
        str(col_y),
        DataFrame(train),
        DataFrame(valid),
        list(params),
    Out:
        H2ORandomForestEstimator(rf)
    Description:
        train h2o random forest model
    '''
    params = [dict(zip(params,t)) for t in zip(*params.values())]

    rf_list = []
    mae_list = []
    for p in params:
        #H2ORandomForestEstimator
        rf = model(**p)
        rf.train(col_X, col_y, training_frame=train, validation_frame=valid)
        mae = rf.mae(valid=True)

        mae_list.append(mae)
        rf_list.append(rf)
        print(mae)

    mae_min, idx = min((val, idx) for (idx, val) in enumerate(mae_list))

    return rf_list[idx]


######## get model prediction func ########
def get_fit_data(model, X, schema):
    '''
    In:
        Any(model),
        DataFrame(X),
        dict(schema),

    Out:
        DataFrame(fit)

    Description:
        fit model and generate submission df
    '''
    h2o_X = h2o.H2OFrame(X, column_types=schema)
    fit = model.predict(h2o_X).as_data_frame()
    fit = fit.assign(Policy_Number = X.index)
    fit = fit.set_index(['Policy_Number'])
    fit.columns = ['Next_Premium']

    return(fit)


######## get model summary func ########
def get_analysis_on_model(model, X, y, fit):
    '''
    In:
        DataFrame(X),
        DataFrame(y),
        DataFrame(fit),

    Out:
        dict(summary)

    Description:
        analyze model output
    '''
    # mae
    mae = (y['Next_Premium'] - fit['Next_Premium']).abs().mean()

    varimp = pd.DataFrame(model.varimp())

    scoring_history = pd.DataFrame(model.scoring_history())

    output = {'mae': mae,
              'varimp': varimp,
              'scoring_history': scoring_history,
              }

    return(output)


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


def read_interim_data(file_name, index_col='Policy_Number'):
    '''
    In: file_name

    Out: interim_data

    Description: read data from directory /data/interim
    '''
    # set the path of raw data
    interim_data_path = os.path.join(os.getcwd(), os.path.pardir, os.path.pardir, 'data', 'interim')

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
    precessed_data_path = os.path.join(os.getcwd(), os.path.pardir, os.path.pardir, 'data', 'processed')
    write_sample_path = os.path.join(precessed_data_path, 'testing-set.csv')
    df.to_csv(write_sample_path)

    return(None)

if __name__ == '__main__':

    # ### Start H2O
    # Start up a 1-node H2O cloud on your local machine, and allow it to use all CPU cores and up to 2GB of memory:

    h2o.init(max_mem_size = "2G")             #specify max number of bytes. uses all cores by default.
    h2o.remove_all()                          #clean slate, in case cluster was already running

    X_train, X_test, y_train, y_test = get_train_input(train_only=True, ext='bs_ext')

    # define model and parameters
    rf_params = {
        'ntrees': [30],
        'max_depth':[15],
        'stopping_metric': ['mae'],
        'stopping_rounds': [2],
        'score_each_iteration': [True],
        'col_sample_rate_per_tree': [1],
        #'sample_rate': [0.4, 0.6, 0.8],
        'seed': [1000000]
    }
    output_rf = train_h2o_model(X_train, X_test, y_train, H2ORandomForestEstimator, rf_params)
    perf_rf_train = get_analysis_on_model(output_rf['model'], X_train, y_train, output_rf['fit_train'])
    perf_rf_test = get_analysis_on_model(output_rf['model'], X_test, y_test, output_rf['fit_test'])

    xg_params = {
        'ntrees': [300],
        #'max_depth':[15] * 3,
        'learn_rate': [0.1],
        'stopping_metric': ['mae'],
        'stopping_rounds': [2],
        'score_each_iteration': [True],
        #'col_sample_rate_per_tree': [0.6, 0.8, 1],
        #'sample_rate': [0.6, 0.8, 1],
        'seed': [1000000]
    }
    output_xg = train_h2o_model(X_train, X_test, y_train, H2OXGBoostEstimator, xg_params)
    perf_xg_train = get_analysis_on_model(output_xg['model'], X_train, y_train, output_xg['fit_train'])
    perf_xg_test = get_analysis_on_model(output_xg['model'], X_test, y_test, output_xg['fit_test'])

    ln_params = {
        'lambda_search': [True],
        'seed': [1000000]
    }
    output_ln = train_h2o_model(X_train, X_test, y_train, H2OGeneralizedLinearEstimator, ln_params)
    perf_ln_train = get_analysis_on_model(output_ln['model'], X_train, y_train, output_ln['fit_train'])
    perf_ln_test = get_analysis_on_model(output_ln['model'], X_test, y_test, output_ln['fit_test'])

    gb_params = {
        'learn_rate': [0.1, 0.2, 0.3],
        'stopping_metric': ['mae'] * 3,
        'stopping_rounds': [2] * 3,
        'score_each_iteration': [True] * 3,
        #'col_sample_rate_per_tree': [0.6, 0.8, 1],
        #'sample_rate': [0.6, 0.8, 1],
        'seed': [1000000] * 3
    }
    output_gb = train_h2o_model(X_train, X_test, y_train, H2OGradientBoostingEstimator, gb_params)
    perf_gb_train = get_analysis_on_model(output_gb['model'], X_train, y_train, output_gb['fit_train'])
    perf_gb_test = get_analysis_on_model(output_gb['model'], X_test, y_test, output_gb['fit_test'])

    h2o.shutdown(prompt=False)