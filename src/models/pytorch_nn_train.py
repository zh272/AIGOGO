import os
import fire
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

## to detach from monitor
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
    loss_fn = F.l1_loss, batch_size = 128, train_params={}, plot=True, 
    test_along=False, optimizer='sgd'
):
    # temp_path = 'feature_distr'
    # if not os.path.isdir(temp_path): os.makedirs(temp_path)
    # for i in range(X_train.values.shape[1]):
    #     idx = np.random.permutation(X_train.values.shape[0])[0:10000]
    #     plt.figure()
    #     plt.scatter(np.random.randn(10000,1), X_train.values[idx, i])
    #     plt.title(X_train.columns[i])
    #     plt.savefig(os.path.join(temp_path, '{}.png'.format(X_train.columns[i][0:30])))
    #     plt.close()

    hyper = {
        'lr':base_lr, 
        'momentum':0.9, 
        'lr_schedule':{
            0:base_lr, 
            max_epoch//4:base_lr/5, 
            max_epoch//2:base_lr/25, 
            max_epoch//4*3:base_lr/125, 
            max_epoch: base_lr/125
        }
    }
    
    
    train_set, X_test_np = get_dataset(X_train.values, y_train.values, X_test.values)

    trainer = Trainer(
        model(**train_params), train_set=train_set, hyper=hyper,
        loss_fn=loss_fn, batch_size=batch_size, epochs=max_epoch, 
        valid_size=0.2, optimizer=optimizer
    )

    valid_hist = []
    for epochs in range(max_epoch):
        trainer.train_epoch()
        if test_along:
            valid_loss = trainer.loss_epoch()
            valid_hist.append(valid_loss)
            print('Epoch {:3}: Training MAE={:.2f}, Valid MAE={:.2f}'.format(epochs, trainer.eval(), valid_loss))
        else:
            print('Epoch {:3}: Training MAE={:.2f}'.format(epochs, trainer.eval()))
    
    state_dict = trainer.model.state_dict()
    if torch.cuda.device_count() > 1:
        input_weights = state_dict['module.regressor.fc0.weight'].cpu().numpy()
    else:
        input_weights = state_dict['regressor.fc0.weight'].cpu().numpy()
    # std_dev = np.std(X_train.values, axis=0) # is 1
    avg_w = np.mean(np.abs(input_weights), axis=0)
    feature_importances = avg_w

    print('====== CatBoost Feature Importances ======')
    feature_names = X_train.columns.values
    sorted_idx = np.argsort(feature_importances*-1) # descending order
    for idx in sorted_idx:
        print('[{:20s}]'.format(feature_names[idx]), '{:7.4f}'.format(feature_importances[idx]))
    
    with open('summary.txt', 'w') as f:
        f.write('====== CatBoost Feature Importances ======\n')
        for idx in sorted_idx:
            f.write('[{:20s}] {:7.4f}\n'.format(feature_names[idx], feature_importances[idx]))
        f.write('\n')

    if plot:
        t_step = np.arange(0, max_epoch, 1)
        train_hist = trainer.evaluator.hist
        fig_path = 'figures'
        if not os.path.isdir(fig_path): os.makedirs(fig_path)
        plt.figure()
        plt.plot(t_step, train_hist, 'r', ls='-', label='training MAE')
        if test_along:
            plt.plot(t_step, valid_hist, 'b', ls='--', label='validation MAE')
        plt.legend(loc='best')
        plt.xlabel('steps')
        plt.title('Training and Validation MAE')
        plt.grid()
        plt.savefig(os.path.join(fig_path, 'training_plot.png'))
        plt.close()

    train_loss = trainer.loss_epoch(load='train')
    valid_loss = trainer.loss_epoch(load='valid')
    print('>>> Final MAE: {:10.4f}(Training), {:10.4f}(Validation)'.format(train_loss,valid_loss))

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

def demo(max_epoch=60, base_lr=0.001, batch_size=128, optimizer='sgd'):
    X_train = read_interim_data('X_train_prefs.csv')
    y_train = read_interim_data('y_train_prefs.csv')
    X_valid = read_interim_data('X_valid_prefs.csv')
    y_valid = read_interim_data('y_valid_prefs.csv')
    X_test = read_interim_data('X_test_prefs.csv')


    feature_list = [
        'int_acc_lia', 'int_claim', 'int_others', 'real_acc_dmg', 'real_acc_lia', 'real_loss', 
        'real_prem_dmg', 'real_prem_ins', 'real_prem_lia', 'real_prem_plc', 'real_prem_thf', 
        'real_prem_vc', 'real_vcost', 'real_ved', 'real_freq_distr', 'real_prem_area_distr', 
        'real_prem_ic_distr', 'real_prem_distr', 'real_prem_ved', 'real_prem_vmm1', 'real_prem_vmm2', 
        'real_prem_vmy', 'real_loss_ins', 'real_prem_ic_nmf_1', 'real_prem_ic_nmf_2', 
        'real_prem_ic_nmf_3', 'real_prem_ic_nmf_4', 'real_prem_ic_nmf_5', 'real_prem_ic_nmf_6', 
        'real_prem_ic_nmf_7', 'real_mc_prob_distr'
    ]


    # Filter features
    X_train = X_train[feature_list]
    X_valid = X_valid[feature_list]
    X_test = X_test[feature_list]

    ### Fill Missing Values
    X_train = X_train.apply(lambda x:x.fillna(-1))
    X_test = X_test.apply(lambda x:x.fillna(-1))

    # begin training
    train_params = {
        'num_input':len(feature_list), 'num_neuron':[60,20,5]
    }
    model_output = get_submission(
        pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]), X_test, 
        model=MLPRegressor, max_epoch=max_epoch, base_lr=base_lr, loss_fn=F.l1_loss, 
        batch_size = batch_size, train_params=train_params, test_along=True, optimizer=optimizer
    )

    # generate submission
    write_precessed_data(model_output['submission'])


if __name__ == '__main__':
    # Example usage: "python nn_train.py --max_epoch 100"
    fire.Fire(demo)