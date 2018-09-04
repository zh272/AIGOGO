import os
import time
import fire
import random
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, auc, roc_curve
from xgboost import XGBClassifier

## to detach from monitor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from helpers import save_obj, load_obj

def get_submission(
    X_train, X_valid, y_train, y_valid, X_test, 
    train_params={}, eval_metric='auc', 
    save=False, load=False, mdl_name='xgb_class'
):
    
    start_time = time.time()
    end_time = start_time
    if load:
        classifier = load_obj(mdl_name)
    else:
        classifier = XGBClassifier(**train_params)
        classifier.fit(X_train.values, y_train.values.ravel(), eval_metric=eval_metric)
        end_time = time.time()
        
        if save:
            save_obj(classifier, mdl_name)
            print('model saved')

    train_pred = classifier.predict(X_train.values)
    valid_pred = classifier.predict(X_valid.values)
    test_pred = classifier.predict(X_test.values)
            
    fpr, tpr, _ = roc_curve(y_train.values, train_pred, pos_label=1)
    train_loss = auc(fpr, tpr)

    fpr, tpr, _ = roc_curve(y_valid.values, valid_pred, pos_label=1)
    valid_loss = auc(fpr, tpr)

    
    feature_importances = classifier.feature_importances_
    
    feature_names = X_train.columns.values
    sorted_idx = np.argsort(feature_importances*-1) # descending order
    
    summary = '====== XGBClassifier Training Summary ======\n'
    for idx in sorted_idx:
        summary += '[{:<25s}] | {:<10.4f}\n'.format(feature_names[idx], feature_importances[idx])
    summary += '>>> training_time={:10.2f}min\n'.format((end_time-start_time)/60)
    summary += '>>> Final AUC: {:10.4f}(Training), {:10.4f}(Validation)\n'.format(train_loss,valid_loss)

    # Generate submission
    submission = pd.DataFrame(data=test_pred,index=X_test.index, columns=['Next_Premium'])

    submission_train = pd.DataFrame(data=train_pred,index=X_train.index, columns=['Next_Premium'])
    
    submission_valid = pd.DataFrame(data=valid_pred,index=X_valid.index, columns=['Next_Premium'])

    return {
        'model': classifier, 'submission': submission, 
        'submission_train':submission_train, 'submission_valid':submission_valid,
        'valid_loss':valid_loss, 'summary':summary
    }


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

    return(interim_data)

def write_precessed_data(df, suffix=None):
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
    if suffix is None:
        file_name = 'testing-set.csv'
    else:
        file_name = 'testing-set_{}.csv'.format(suffix)
    write_sample_path = os.path.join(precessed_data_path, file_name)
    df.to_csv(write_sample_path)


# empirical scale: weight_decay=0.0001
def demo(
    epochs=300, base_lr=0.1, max_depth=5, subsample=0.8, 
    objective='binary:logistic', eval_metric='auc', tree_method='gpu_exact',
    colsample_bytree=0.8, colsample_bylevel=0.8, gamma=0.0, 
    reg_alpha=3.0, reg_lambda=0.0, max_delta_step=0, 
    get_train=False, get_test=False, save=False, load=False, seed=None
):
    if seed is not None:
        rand_reset(seed)
    X_train = read_interim_data('X_train_prefs.csv')
    y_train = read_interim_data('y_train_prefs.csv')
    X_valid = read_interim_data('X_valid_prefs.csv')
    y_valid = read_interim_data('y_valid_prefs.csv')
    X_test = read_interim_data('X_test_prefs.csv')

    y_train.loc[y_train['Next_Premium'] != 0, 'Next_Premium'] = 1
    y_valid.loc[y_valid['Next_Premium'] != 0, 'Next_Premium'] = 1


    feature_list = [feature for feature in X_train.columns.values if 'cat_' not in feature]
    num_features = len(feature_list)
    
    print('Number of features: {}'.format(num_features))

    num_1 = y_train['Next_Premium'].sum()
    num_0 = len(y_train) - num_1
    print('Class 0: # {}'.format(num_0))
    print('Class 1: # {}'.format(num_1))


    # Filter features
    X_train = X_train[feature_list]
    X_valid = X_valid[feature_list]
    X_test = X_test[feature_list]

    ### Fill Missing Values
    X_train = X_train.apply(lambda x:x.fillna(-1))
    X_valid = X_valid.apply(lambda x:x.fillna(-1))
    X_test = X_test.apply(lambda x:x.fillna(-1))

    # begin training
    # scale = num_0/num_1
    # scale = num_1/num_0
    scale = 1
    train_params = {
        'n_estimators': epochs, 'learning_rate': base_lr, 
        'objective':objective, 'max_delta_step':max_delta_step, 
        'max_depth':max_depth, 'subsample':subsample, 
        'colsample_bytree':colsample_bytree, 'colsample_bylevel':colsample_bylevel, 
        'gamma':gamma, 'reg_alpha':reg_alpha, 'reg_lambda':reg_lambda, 
        'scale_pos_weight':scale, 'tree_method':tree_method
    }
    
    model_output = get_submission(
        X_train, X_valid, y_train, y_valid, X_test,
        eval_metric=eval_metric, train_params=train_params, 
        save=save, load=load
    )

    summary = model_output['summary']
    summary += '>>> random seed: {}\n'.format(seed)
    
    # y_train_pred = model_output['submission_train'].data
    y_valid_pred = model_output['submission_valid'].values
    
    summary += '{}\n'.format(confusion_matrix(y_valid, y_valid_pred))

    print(summary)

    # generate submission
    if get_test:
        write_precessed_data(model_output['submission'], suffix='mlptest{}'.format(int(model_output['valid_loss'])))
        with open('summary_mlp{}.txt'.format(int(model_output['valid_loss'])), 'w') as f:
            f.write(summary)

    if get_train:
        write_precessed_data(model_output['submission_train'], suffix='mlptrain')
        write_precessed_data(model_output['submission_valid'], suffix='mlpvalid')

def rand_reset(seed):
    random.seed(seed)
    torch.manual_seed(random.randint(0,1000))
    torch.cuda.manual_seed_all(random.randint(0,1000))
    np.random.seed(random.randint(0,1000))

if __name__ == '__main__':
    # Example usage: "python nn_train.py --epochs 100"
    fire.Fire(demo)