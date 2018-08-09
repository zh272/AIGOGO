import os
import fire
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from trainer import Trainer
from model import MLPRegressor
from helpers import get_dataset, test_epoch, ready

import warnings
warnings.simplefilter("ignore", UserWarning)

def get_submission(
    X_train, y_train, X_test, model=MLPRegressor, max_epoch=200, base_lr=0.1, 
    loss_fn = F.l1_loss, batch_size = 128, train_params={}, plot=True
):
    hyper = {
        'lr':base_lr, 
        'momentum':0.9, 
        'lr_schedule':{
            0:base_lr, 
            max_epoch//2:base_lr/5, 
            max_epoch//4*3:base_lr/25, 
            max_epoch: base_lr/125
        }
    }

    train_set, X_test_np = get_dataset(X_train.values, y_train.values, X_test.values)

    trainer = Trainer(
        model(**train_params), train_set=train_set, hyper=hyper,
        loss_fn=loss_fn, batch_size=batch_size, epochs=max_epoch
    )


    for epochs in range(max_epoch):
        trainer.train_epoch()
        if ready(epochs, threshold=1):
            print('Epoch {:3}: Training MAE={:.2f}'.format(epochs, trainer.eval()))

    if plot:
        t_step = np.arange(0, max_epoch, 1)
        train_hist = trainer.evaluator.hist
        fig_path = 'figures'
        if not os.path.isdir(fig_path): os.makedirs(fig_path)
        plt.figure()
        plt.plot(t_step, train_hist, 'r', ls='-', label='training MAE()')
        plt.legend(loc='best')
        plt.xlabel('steps')
        plt.title('Training MAE')
        plt.grid()
        plt.savefig(os.path.join('figures', 'training_plot.png'))
        plt.close()

    valid_loss = trainer.loss_epoch()
    print('>>> Validation MAE: {:10.4f}'.format(valid_loss))

    # Generate submission
    test_output = trainer.predict(torch.FloatTensor(X_test_np)).cpu().data.numpy()
    submission = pd.DataFrame(data=test_output,index=X_test.index, columns=['Next_Premium'])

    return {'model': trainer, 'submission': submission}


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

def demo(max_epoch=50, base_lr=0.1, batch_size=128, loss_fn=F.l1_loss):
    X_train = read_interim_data('X_train_id.csv')
    X_test = read_interim_data('X_test_id.csv')
    y_train = read_interim_data('y_train.csv')

    num_features = [
        'ipolicy_coverage_avg', 'ipolicy_premium_avg', 'ivehicle_repcost_avg',
        'ipolicies', 'iclaims', 'iclaim_paid_amount', 'iclaim_salvage_amount', 
        'cpremium_dmg', 'cpremium_lia', 'cpremium_thf', 
        'Multiple_Products_with_TmNewa_(Yes_or_No?)', 'cclaims', 'closs', 'csalvate', 
    ]
    encodeCols = []
    # encodeCols = [
    #     "Insured's_ID", 'fmarriage', 'fsex', 'iage_lab',
    #     'iassured_lab',  'Vehicle_identifier',
    #     'Coding_of_Vehicle_Branding_&_Type', 'Vehicle_Make_and_Model1',
    #     'Vehicle_Make_and_Model2', 'vlocomotive', 'vyear_lab', 'vregion_lab',
    #     'vengine_lab', 'cbucket', 'aassured_zip', 'iply_area', 'Distribution_Channel',
    #     'cclaim_id', 'ccause_type'
    # ]

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

    X_train['vyear_lab'] = X_train['vyear_lab'].round().astype(int)
    X_test['vyear_lab'] = X_test['vyear_lab'].round().astype(int)

    # Filter features
    X_train = X_train[feature_list]
    X_test = X_test[feature_list]

    ### Fill Missing Values
    # X_train[num_features] = X_train[num_features].apply(lambda x:x.fillna(x.value_counts().index[0]))
    X_train = X_train.apply(lambda x:x.fillna(-1))
    X_test = X_test.apply(lambda x:x.fillna(-1))

    # begin training
    train_params = {
        'num_input':len(feature_list), 'num_neuron':[100,25,5]
    }
    model_output = get_submission(
        X_train, y_train, X_test, model=MLPRegressor, max_epoch=max_epoch, 
        base_lr=base_lr, loss_fn = loss_fn, batch_size = batch_size, train_params=train_params
    )

    # generate submission
    write_precessed_data(model_output['submission'])


if __name__ == '__main__':
    # Example usage: "python nn_train.py --max_epoch 100"
    fire.Fire(demo)