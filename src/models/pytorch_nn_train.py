import os
import time
import fire
import torch
import random
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
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
    momentum=0.9, weight_decay=0.0001, batch_size = 128, train_params={}, plot=True, zero_predict=False,
    test_along=False, optimizer='sgd', hyper={}, save=False, load=False, mdl_name='mlp.pt',all_train=False
):    
    if all_train:
        X_train = pd.concat([X_train, X_valid])
        y_train = pd.concat([y_train, y_valid])
        X_valid = None
        y_valid = None
        test_along = False

        train_set, valid_set, X_test_np, X_train_np, X_valid_np = get_dataset(
            X_train.values, y_train.values, X_test.values
        )
    else:
        train_set, valid_set, X_test_np, X_train_np, X_valid_np = get_dataset(
            X_train.values, y_train.values, X_test.values, X_valid.values, y_valid.values
        )
    
    PATH = './saved_models'
    if not os.path.isdir(PATH): os.makedirs(PATH)
    
    start_time = time.time()
    end_time = start_time
    if load:
        trainer = Trainer(
            torch.load(os.path.join(PATH, mdl_name)), train_set=train_set, loss_fn=F.l1_loss, hyper=hyper,
            valid_set=valid_set, batch_size=batch_size, epochs=max_epoch, optimizer=optimizer
        )

    else:
        trainer = Trainer(
            model(**train_params), train_set=train_set, loss_fn=F.l1_loss, hyper=hyper,
            valid_set=valid_set, batch_size=batch_size, epochs=max_epoch, optimizer=optimizer
        )

        valid_hist = []
        for epochs in range(max_epoch):
            trainer.train_epoch()

            temp_lr = trainer.optimizer.param_groups[0]['lr']
            
            if test_along:
                temp_valid = trainer.loss_epoch()
                valid_hist.append(temp_valid)
                print('Epoch {:3}: Training MAE={:8.2f}, Valid MAE={:8.2f}, lr={}'.format(epochs, trainer.eval(load='train'), temp_valid, temp_lr))
            else:
                print('Epoch {:3}: Training MAE={:8.2f}, lr={}'.format(epochs, trainer.eval(load='train'), temp_lr))
        end_time = time.time()
        
        
        
        if plot:
            t_step = np.arange(0, max_epoch, 1)
            train_hist = trainer.evaluator_loss.hist
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

        if save:
            torch.save(trainer.model, os.path.join(PATH, mdl_name))

    # train_loss = trainer.loss_epoch(load='train')
    # valid_loss = trainer.loss_epoch(load='valid')
    
    train_pred = trainer.predict(torch.FloatTensor(X_train_np)).cpu().data.numpy()
    train_loss = mean_absolute_error(y_train.values, train_pred)
    
    if X_valid_np is not None:
        valid_pred = trainer.predict(torch.FloatTensor(X_valid_np)).cpu().data.numpy()
        valid_loss = mean_absolute_error(y_valid.values, valid_pred)
    else:
        valid_pred = None
        valid_loss = 0

    test_pred = trainer.predict(torch.FloatTensor(X_test_np)).cpu().data.numpy()

    # print('>>> original valid loss: {}'.format(valid_loss))
    # valid_pred[valid_pred<=50] = 0
    # valid_pred[np.absolute(valid_pred-100)<50] = 100
    # new_valid_loss = mean_absolute_error(y_valid.values, valid_pred)
    # print('>>> New valid loss: {}'.format(new_valid_loss))



    if zero_predict:
        # zero_predictor = load_obj('xgb_class')
        # valid_pred_zeros = zero_predictor.predict(X_valid.values)

        zero_predictor = torch.load(os.path.join(PATH, 'mlp_class.pt'))
        zero_predictor.eval() # evaluation mode
        inp = torch.FloatTensor(X_valid.values)
        with torch.no_grad():
            if torch.cuda.is_available():
                inp = torch.autograd.Variable(inp.cuda())
            else:
                inp = torch.autograd.Variable(inp)
        valid_pred_zeros = zero_predictor(inp).cpu().data.numpy().argsort(axis=1)[:,-1]

        # train_pred_zeros = zero_predictor.predict(X_train.values)
        # train_output[train_pred_zeros == 0] = 0
        print('>>> original valid loss: {}'.format(valid_loss))
        valid_pred[valid_pred_zeros==0,:] = 0

        new_valid_loss = mean_absolute_error(y_valid.values, valid_pred)
        print('>>> New valid loss: {}'.format(new_valid_loss))
    

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
    
    summary = '====== MLPRegressor Training Summary ======\n'
    summary += '>>> epochs={}, lr={}, momentum={}, weight_decay={}\n'.format(max_epoch,base_lr,momentum,weight_decay)
    summary += '>>> schedule={}\n'.format(hyper['lr_schedule']) if 'lr_schedule' in hyper else 'None'
    summary += '>>> hidden={}, optimizer="{}", batch_size={}\n'.format(train_params['num_neuron'],optimizer,batch_size)
    for idx in sorted_idx:
        summary += '[{:<30s}] {:<10.4f}\n'.format(feature_names[idx], feature_importances[idx])
    summary += '>>> training_time={:10.2f}min\n'.format((end_time-start_time)/60)
    summary += '>>> Final MAE: {:10.4f}(Training), {:10.4f}(Validation)\n'.format(train_loss,valid_loss)

    # Generate submission
    submission = pd.DataFrame(data=test_pred,index=X_test.index, columns=['Next_Premium'])
    
    submission_train = pd.DataFrame(data=train_pred,index=X_train.index, columns=['Next_Premium'])
    
    submission_valid = pd.DataFrame(data=valid_pred,index=X_valid.index, columns=['Next_Premium']) if valid_pred is not None else None


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
    epochs=100, base_lr=0.001, momentum=0.9, weight_decay=0, all_train=False,
    batch_size=128, optimizer='sgd', dropout=False, seed=random.randint(0,1000), 
    get_train=False, get_test=False, save=False, load=False
):
    rand_reset(seed)
    X_train = read_interim_data('X_train_new.csv')
    y_train = read_interim_data('y_train_new.csv')
    X_valid = read_interim_data('X_valid_new.csv')
    y_valid = read_interim_data('y_valid_new.csv')
    X_test = read_interim_data('X_test_new.csv')

    feature_list = [feature for feature in X_train.columns.values if 'cat_' not in feature]

    # feature_list = [feature for feature in feature_list if 'vequip' not in feature]
    # feature_list = [feature for feature in feature_list if 'nmf' not in feature]
    # feature_list = [feature for feature in feature_list if 'real_mc_mean_diff' in feature or 'real_acc' in feature]
    feature_list = [feature for feature in feature_list if 'nmf' in feature]


    num_features = len(feature_list)
    print('Number of features: {}'.format(num_features))

    # Filter features
    X_train = X_train[feature_list]
    X_valid = X_valid[feature_list]
    X_test = X_test[feature_list]

    ### Fill Missing Values
    X_train = X_train.apply(lambda x:x.fillna(-1))
    X_valid = X_valid.apply(lambda x:x.fillna(-1))
    X_test = X_test.apply(lambda x:x.fillna(-1))

    # begin training
    # num_neuron = [100,50,8]
    num_neuron = [20,10]
    print('Network Architecture: {}'.format(num_neuron))
    # num_neuron = [round(1.5*num_features),round(0.3*num_features),round(0.1*num_features)]
    # num_neuron = [160,30,8]

    train_params = {
        'num_input':len(feature_list), 'dropout':dropout, 
        'num_neuron':num_neuron
    }
    optim_hyper = {
        'lr':base_lr, 
        'momentum':momentum,
        'weight_decay':weight_decay, 
        # 'scheduler': 'plateau',
        'lr_schedule':{
            10:base_lr,
            25:base_lr/2, 
            50:base_lr/10, 
            100:base_lr/100,
            200:base_lr/1000
        }
    }

    # optim_hyper = {
    #     'lr':base_lr, 
    #     'momentum':momentum,
    #     'weight_decay':weight_decay, 
    #     'lr_schedule':{
    #         epochs//4:base_lr,
    #         epochs//2:base_lr/5, 
    #         epochs//4*3:base_lr/50, 
    #         epochs: base_lr/200
    #     }
    # }
    
    model_output = get_submission(
        X_train, X_valid, y_train, y_valid, X_test, all_train=all_train,
        model=MLPRegressor, max_epoch=epochs, base_lr=base_lr, 
        momentum=momentum, weight_decay=weight_decay,
        batch_size = batch_size, train_params=train_params, 
        test_along=True, optimizer=optimizer, hyper=optim_hyper,
        save=save, load=load, mdl_name='mlp.pt'
    )

    summary = model_output['summary']
    summary += '>>> random seed: {}\n'.format(seed)

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