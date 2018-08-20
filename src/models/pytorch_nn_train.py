import os
import time
import fire
import torch
import random
import numpy as np
import pandas as pd
import torch.nn.functional as F

## to detach from monitor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from trainer import Trainer
from model import MLPRegressor
from helpers import get_dataset, test_epoch, ready, save_obj, load_obj

def get_submission(
    X_train, X_valid, y_train, y_valid, X_test, model=MLPRegressor, max_epoch=200, base_lr=0.1, 
    momentum=0.9, weight_decay=0.0001, batch_size = 128, train_params={}, plot=True, 
    test_along=False, optimizer='sgd', hyper={}, save=False
):    
    train_set, valid_set, X_test_np = get_dataset(
        X_train.values, y_train.values, X_test.values, X_valid.values, y_valid.values
    )

    trainer = Trainer(
        model(**train_params), train_set=train_set, loss_fn=F.l1_loss, hyper=hyper,
        valid_set=valid_set, batch_size=batch_size, epochs=max_epoch, optimizer=optimizer
    )

    valid_hist = []
    start_time = time.time()
    for epochs in range(max_epoch):
        trainer.train_epoch()
        if test_along:
            temp_valid = trainer.loss_epoch()
            valid_hist.append(temp_valid)
            print('Epoch {:3}: Training MAE={:.2f}, Valid MAE={:.2f}'.format(epochs, trainer.eval(), temp_valid))
        else:
            print('Epoch {:3}: Training MAE={:.2f}'.format(epochs, trainer.eval()))
    end_time = time.time()
    
    state_dict = trainer.model.state_dict()
    if torch.cuda.device_count() > 1:
        input_weights = state_dict['module.regressor.fc0.weight'].cpu().numpy()
    else:
        input_weights = state_dict['regressor.fc0.weight'].cpu().numpy()
    # assume std deviation of each feature is 1
    avg_w = np.mean(np.abs(input_weights), axis=0)
    feature_importances = avg_w
    
    feature_names = X_train.columns.values
    sorted_idx = np.argsort(feature_importances*-1) # descending order
    
    train_loss = trainer.loss_epoch(load='train')
    valid_loss = trainer.loss_epoch(load='valid')

    summary = '====== MLPRegressor Training Summary ======\n'
    summary += '>>> epochs={}, lr={}, momentum={}, weight_decay={}\n'.format(max_epoch,base_lr,momentum,weight_decay)
    summary += '>>> schedule={}\n'.format(hyper['lr_schedule'])
    summary += '>>> hidden={}, optimizer="{}", batch_size={}\n'.format(train_params['num_neuron'],optimizer,batch_size)
    for idx in sorted_idx:
        summary += '[{:25s}] {:10.4f}\n'.format(feature_names[idx], feature_importances[idx])
    summary += '>>> training_time={:.2f}min\n'.format((end_time-start_time)/60)
    summary += '>>> Final MAE: {:10.4f}(Training), {:10.4f}(Validation)\n'.format(train_loss,valid_loss)

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


    # Generate submission
    test_output = trainer.predict(torch.FloatTensor(X_test_np)).cpu().data.numpy()
    submission = pd.DataFrame(data=test_output,index=X_test.index, columns=['Next_Premium'])

    if save:
        save_obj({'MLPRegressor':trainer.model}, 'mlp_model')
        # To load model, use:
        # model = load_obj('mlp_model')['MLPRegressor']

    return {'model': trainer, 'submission': submission, 'valid_loss':valid_loss, 'summary':summary}


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
    if isinstance(suffix, float) or isinstance(suffix, int):
        file_name = 'testing-set_{}.csv'.format(int(suffix))
    else:
        file_name = 'testing-set.csv'
    write_sample_path = os.path.join(precessed_data_path, file_name)
    df.to_csv(write_sample_path)

    return(None)

# empirical scale: weight_decay=0.0001
def demo(epochs=80, base_lr=0.001, momentum=0.8, weight_decay=0, batch_size=128, optimizer='sgd', seed=None):
    if seed is not None:
        # known best seed=10
        rand_reset(seed)
    X_train = read_interim_data('X_train_prefs.csv')
    y_train = read_interim_data('y_train_prefs.csv')
    X_valid = read_interim_data('X_valid_prefs.csv')
    y_valid = read_interim_data('y_valid_prefs.csv')
    X_test = read_interim_data('X_test_prefs.csv')

    feature_list = [feature for feature in X_train.columns.values if 'cat_' not in feature]


    # Filter features
    X_train = X_train[feature_list]
    X_valid = X_valid[feature_list]
    X_test = X_test[feature_list]

    ### Fill Missing Values
    X_train = X_train.apply(lambda x:x.fillna(-1))
    X_test = X_test.apply(lambda x:x.fillna(-1))

    # begin training
    train_params = {
        'num_input':len(feature_list), 'num_neuron':[110,30,6]
    }
    optim_hyper = {
        'lr':base_lr, 
        'momentum':momentum,
        'weight_decay':weight_decay, 
        'lr_schedule':{
            0:base_lr, 
            epochs//4:base_lr/5, 
            epochs//2:base_lr/25, 
            epochs//4*3:base_lr/125, 
            epochs: base_lr/125
        }
    }
    # optim_hyper = {
    #     'lr':base_lr, 
    #     'momentum':momentum,
    #     'weight_decay':weight_decay, 
    #     'lr_schedule':{
    #         0:base_lr, 
    #         epochs//2:base_lr/10, 
    #         epochs//4*3:base_lr/100, 
    #         epochs: base_lr/100
    #     }
    # }
    model_output = get_submission(
        X_train, X_valid, y_train, y_valid, X_test, 
        model=MLPRegressor, max_epoch=epochs, base_lr=base_lr, momentum=momentum, weight_decay=weight_decay,
        batch_size = batch_size, train_params=train_params, test_along=True, optimizer=optimizer, hyper=optim_hyper
    )

    summary = model_output['summary']
    summary += '>>> random seed: {}\n'.format(seed)

    print(summary)
    with open('summary_{}.txt'.format(int(model_output['valid_loss'])), 'w') as f:
        f.write(summary)

    # generate submission
    write_precessed_data(model_output['submission'], suffix=model_output['valid_loss'])

def rand_reset(seed):
    random.seed(seed)
    torch.manual_seed(random.randint(0,1000))
    torch.cuda.manual_seed_all(random.randint(0,1000))
    np.random.seed(random.randint(0,1000))

if __name__ == '__main__':
    # Example usage: "python nn_train.py --epochs 100"
    fire.Fire(demo)