import os
import time
import fire
import torch
import random
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F

## to detach from monitor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from trainer import Trainer
from model import MLPClassifier
from helpers import get_dataset, test_epoch, ready, save_obj, load_obj

def get_submission(
    X_train, X_valid, y_train, y_valid, X_test, model=MLPClassifier, max_epoch=200, base_lr=0.1, 
    momentum=0.9, weight_decay=0.0001, batch_size = 128, train_params={}, plot=True, 
    test_along=False, optimizer='sgd', hyper={}, save=False, load=False, mdl_name='mlp_class.pt'
):    
    train_set, valid_set, X_test_np, X_train_np, X_valid_np = get_dataset(
        X_train.values, y_train.values, X_test.values, X_valid.values, y_valid.values,
        target_type=torch.LongTensor, target_shape=(-1,)
    )
    
    PATH = './saved_models'
    if not os.path.isdir(PATH): os.makedirs(PATH)
    
    start_time = time.time()
    end_time = start_time
    if load:
        trainer = Trainer(
            torch.load(os.path.join(PATH, mdl_name)), train_set=train_set, loss_fn=F.cross_entropy, hyper=hyper,
            valid_set=valid_set, batch_size=batch_size, epochs=max_epoch, optimizer=optimizer
        )
    else:
        trainer = Trainer(
            model(**train_params), train_set=train_set, loss_fn=F.cross_entropy, hyper=hyper,
            valid_set=valid_set, batch_size=batch_size, epochs=max_epoch, optimizer=optimizer
        )

        valid_hist = []
        for epochs in range(max_epoch):
            trainer.train_epoch()

            temp_lr = trainer.optimizer.param_groups[0]['lr']
            
            if test_along:
                temp_valid = trainer.loss_epoch(eval_type='err')
                valid_hist.append(temp_valid)
                print('Epoch {:3}: Training ERR={:8.2f}, Valid ERR={:8.2f}, lr={}'.format(epochs, trainer.eval(eval_type='err')*100, temp_valid*100, temp_lr))
            else:
                print('Epoch {:3}: Training ERR={:8.2f}, lr={}'.format(epochs, trainer.eval(eval_type='err')*100, temp_lr))
        end_time = time.time()
        
        
        
        if plot:
            t_step = np.arange(0, max_epoch, 1)
            train_hist = trainer.evaluator_err.hist
            fig_path = 'figures'
            if not os.path.isdir(fig_path): os.makedirs(fig_path)
            plt.figure()
            plt.plot(t_step, train_hist, 'r', ls='-', label='training ERR')
            if test_along:
                plt.plot(t_step, valid_hist, 'b', ls='--', label='validation ERR')
            plt.legend(loc='best')
            plt.xlabel('steps')
            plt.title('Training and Validation ERR')
            plt.grid()
            plt.savefig(os.path.join(fig_path, 'training_plot.png'))
            plt.close()

        if save:
            torch.save(trainer.model, os.path.join(PATH, mdl_name))

            
    train_loss = trainer.loss_epoch(load='train', eval_type='err')
    valid_loss = trainer.loss_epoch(load='valid', eval_type='err')

    
    state_dict = trainer.model.state_dict()
    if torch.cuda.device_count() > 1:
        input_weights = state_dict['module.classifier.fc0.weight'].cpu().numpy()
    else:
        input_weights = state_dict['classifier.fc0.weight'].cpu().numpy()
    # assume std deviation of each feature is 1
    avg_w = np.mean(np.abs(input_weights), axis=0)
    feature_importances = avg_w
    
    feature_names = X_train.columns.values
    sorted_idx = np.argsort(feature_importances*-1) # descending order
    
    summary = '====== MLPClassifier Training Summary ======\n'
    summary += '>>> epochs={}, lr={}, momentum={}, weight_decay={}\n'.format(max_epoch,base_lr,momentum,weight_decay)
    summary += '>>> schedule={}\n'.format(hyper['lr_schedule'])
    summary += '>>> hidden={}, optimizer="{}", batch_size={}\n'.format(train_params['num_neuron'],optimizer,batch_size)
    for idx in sorted_idx:
        summary += '[{:<25s}] {:<10.4f}\n'.format(feature_names[idx], feature_importances[idx])
    summary += '>>> training_time={:10.2f}min\n'.format((end_time-start_time)/60)
    summary += '>>> Final ERR: {:10.4f}(Training), {:10.4f}(Validation)\n'.format(train_loss,valid_loss)

    # Generate submission
    test_output = trainer.predict(torch.FloatTensor(X_test_np)).cpu().data.numpy().argsort(axis=1)[:,-1]
    submission = pd.DataFrame(data=test_output,index=X_test.index, columns=['Next_Premium'])
    
    train_output = trainer.predict(torch.FloatTensor(X_train_np)).cpu().data.numpy().argsort(axis=1)[:,-1]
    submission_train = pd.DataFrame(data=train_output,index=X_train.index, columns=['Next_Premium'])
    
    valid_output = trainer.predict(torch.FloatTensor(X_valid_np)).cpu().data.numpy().argsort(axis=1)[:,-1]
    submission_valid = pd.DataFrame(data=valid_output,index=X_valid.index, columns=['Next_Premium'])

    return {
        'model': trainer, 'submission': submission, 
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
    epochs=100, base_lr=0.001, momentum=0.9, weight_decay=0, 
    batch_size=128, optimizer='sgd', dropout=False, seed=None, 
    get_train=False, get_test=False, save=False, load=False
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
    num_neuron = [160,60,20,5]
    # num_neuron = [round(1.5*num_features),round(0.3*num_features),round(0.1*num_features)]
    # num_neuron = [160,30,8]

    train_params = {
        'num_input':len(feature_list), 'dropout':dropout, 
        'num_neuron':num_neuron, 'num_class': 2
    }

    optim_hyper = {
        'lr':base_lr, 
        'momentum':momentum,
        'weight_decay':weight_decay, 
        'lr_schedule':{
            epochs//4:base_lr,
            epochs//2:base_lr/5, 
            epochs//4*3:base_lr/50, 
            epochs: base_lr/200
        }
    }
    
    model_output = get_submission(
        X_train, X_valid, y_train, y_valid, X_test, 
        model=MLPClassifier, max_epoch=epochs, base_lr=base_lr, 
        momentum=momentum, weight_decay=weight_decay,
        batch_size = batch_size, train_params=train_params, 
        test_along=True, optimizer=optimizer, hyper=optim_hyper,
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