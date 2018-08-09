import h2o
import os
import pandas as pd
from tabulate import tabulate
from sklearn.metrics import confusion_matrix
from h2o.estimators.random_forest import H2ORandomForestEstimator

import warnings
warnings.simplefilter("ignore", UserWarning)

def get_submission(X_train, y_train, X_test, params, cate_col=None, col_types=None):
    '''
    In:
        H2OFrame(h2o_sample)
    Out:
        Any(rf_v1) -> random forest model
    Description:
        basic random forest model for submission
    '''
    df_train = y_train.merge(X_train, how='left', left_index=True, right_index=True)
    h2o_train = h2o.H2OFrame(df_train, column_types=col_types)
    
    # df_test = y_test.merge(X_test, how='left', left_index=True, right_index=True)
    h2o_test = h2o.H2OFrame(X_test, column_types=col_types)

    # if cate_col:
    #     for cate in cate_col:
    #         h2o_train[cate].asfactor()
    #         h2o_test[cate].asfactor()

    # split train into train and valid
    train, valid = h2o_train.split_frame(ratios = [0.8], seed=0)

    # separate independent variables from dependent variables
    col_y = 'Next_Premium'
    col_X = list(X_train.columns)

    # create model
    regressor = H2ORandomForestEstimator(**params)
    regressor.train(col_X, col_y, training_frame=train, validation_frame=valid)

    train_output = regressor.predict(train[col_X])
    valid_output = regressor.predict(valid[col_X])

    train_target = train[col_y]
    valid_target = valid[col_y]

    # print('=== Train Confusion matrix ===')
    # print(confusion_matrix(train_target.as_data_frame().values, train_output.as_data_frame().values[:,0]))
    # print('=== Valid Confusion matrix ===')
    # print(confusion_matrix(valid_target.as_data_frame().values, valid_output.as_data_frame().values[:,0]))


    training_mae = (train_output - train_target).abs().mean()
    valid_mae = (valid_output - valid_target).abs().mean()
    
    fea_imp = regressor.varimp(use_pandas=True)
    print('====== Feature Importances ======')
    print(fea_imp)
    print('training MAE:{}'.format(training_mae))
    print('validation MAE:{}'.format(valid_mae))
    
    content = tabulate(fea_imp.values.tolist(), list(fea_imp.columns), tablefmt="plain")
    with open('summary.txt', 'w') as f:
        f.write('====== Feature Importances ======\n')
        # fea_imp.to_csv(f, header=False)
        f.write(content)
        f.write('\n')
        f.write('training MAE:{}\n'.format(training_mae))
        f.write('validation MAE:{}\n'.format(valid_mae))


    submission = regressor.predict(h2o_test[col_X]).as_data_frame()
    submission['Policy_Number'] = X_test.index
    submission = submission.set_index(['Policy_Number'])
    submission.columns = ['Next_Premium']

    return({'model': regressor, 'submission': submission})


def read_interim_data(file_name, index_col='Policy_Number'):
    '''
    In: file_name
    Out: interim_data
    Description: read data from directory /data/interim
    '''
    # set the path of raw data
    interim_data_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), os.path.pardir, os.path.pardir, 'data', 'interim'
    )

    file_path = os.path.join(interim_data_path, file_name)
    interim_data = pd.read_csv(file_path, index_col=index_col)

    # interim_data = interim_data.apply(lambda x:x.fillna(x.value_counts().index[0]))


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
    precessed_data_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), os.path.pardir, os.path.pardir, 'data', 'processed'
    )
    write_sample_path = os.path.join(precessed_data_path, 'testing-set.csv')
    df.to_csv(write_sample_path)

    return(None)

if __name__ == '__main__':

    # ### Start H2O
    # Start up a 1-node H2O cloud on your local machine, and allow it to use all CPU cores and up to 2GB of memory:

    h2o.init(ip="localhost", port=54323, max_mem_size = "2G")             #specify max number of bytes. uses all cores by default.
    h2o.remove_all()                          #clean slate, in case cluster was already running

    X_train = read_interim_data('X_train_id.csv')
    X_test = read_interim_data('X_test_id.csv')
    y_train = read_interim_data('y_train.csv')
    # y_test = read_interim_data('testing-set.csv')

    num_features = [
        'ipolicy_coverage_avg', 'ipolicy_premium_avg', 'ivehicle_repcost_avg',
        'ipolicies', 'iclaims', 'iclaim_paid_amount', 'iclaim_salvage_amount', 
        'cpremium_dmg', 'cpremium_lia', 'cpremium_thf', 
        'Multiple_Products_with_TmNewa_(Yes_or_No?)', 'cclaims', 'closs', 'csalvate', 
    ]
    encodeCols = [
        "Insured's_ID", 'fmarriage', 'fsex', 'iage_lab',
        'iassured_lab',  'Vehicle_identifier',
        'Coding_of_Vehicle_Branding_&_Type', 'Vehicle_Make_and_Model1',
        'Vehicle_Make_and_Model2', 'vlocomotive', 'vyear_lab', 'vregion_lab',
        'vengine_lab', 'cbucket', 'aassured_zip', 'iply_area', 'Distribution_Channel',
        'cclaim_id', 'ccause_type'
    ]

    cat_features = []
    print('===== {:20s} ====='.format('Disgarded Features'))
    for name in encodeCols:
        variance = X_train[name].unique()
        # ignore low distinct features
        # threshold could be changed
        if len(variance) > 1 and len(variance) < len(X_train):
            cat_features.append(name)
        else:
            print('[{:50s}]'.format(name),  len(variance))

    feature_list = cat_features + num_features

    
    col_types = {}
    for fea in num_features:
        col_types[fea] = 'real'
    for fea in cat_features:
        col_types[fea] = 'enum'

    # X_train[num_features] = X_train[num_features].apply(lambda x:x.fillna(x.value_counts().index[0]))
    X_train['vyear_lab'] = X_train['vyear_lab'].round().astype(int)
    X_test['vyear_lab'] = X_test['vyear_lab'].round().astype(int)

    ### Fill Missing Values
    # X_train[num_features] = X_train[num_features].apply(lambda x:x.fillna(-1))
    # X_test[num_features] = X_test[num_features].apply(lambda x:x.fillna(-1))
    # X_train[cat_features] = X_train[cat_features].apply(lambda x:x.fillna('NaNa')).astype(str)
    # X_test[cat_features] = X_test[cat_features].apply(lambda x:x.fillna('NaNa')).astype(str)

    X_train = X_train[feature_list]
    X_test = X_test[feature_list]

    # default ntrees=50, max_depth=20, stopping_rounds=0
    params = {
        'model_id':"rf_v1",
        'ntrees':500, 'max_depth':15, 'stopping_metric':'mae',
        'stopping_rounds':2, 'score_each_iteration':True,
        'seed':1000000
    }

    # y_train[y_train>0] = 1
    # col_types['Next_Premium'] = 'categorical'

    model_output = get_submission(X_train, y_train, X_test, params, cate_col=cat_features, col_types=col_types)
    # write_precessed_data(model_output['submission'])

    h2o.cluster().shutdown()