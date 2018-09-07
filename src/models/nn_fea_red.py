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
    test_along=False, optimizer='sgd', hyper={}, save=False, load=False, mdl_name='mlp_fea_red.pt'
):
    train_set, valid_set, X_test_np, X_train_np, X_valid_np, scaler = get_dataset(
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
                print('Epoch {:3}: Training MAE={:8.2f}, Valid MAE={:8.2f}, lr={}'.format(epochs, trainer.eval(), temp_valid, temp_lr))
            else:
                print('Epoch {:3}: Training MAE={:8.2f}, lr={}'.format(epochs, trainer.eval(), temp_lr))
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

            
    train_loss = trainer.loss_epoch(load='train')
    valid_loss = trainer.loss_epoch(load='valid')

    
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
    summary += '>>> schedule={}\n'.format(hyper['lr_schedule'])
    summary += '>>> hidden={}, optimizer="{}", batch_size={}\n'.format(train_params['num_neuron'],optimizer,batch_size)
    for idx in sorted_idx:
        summary += '[{:<25s}] {:<10.4f}\n'.format(feature_names[idx], feature_importances[idx])
    summary += '>>> training_time={:10.2f}min\n'.format((end_time-start_time)/60)
    summary += '>>> Final MAE: {:10.4f}(Training), {:10.4f}(Validation)\n'.format(train_loss,valid_loss)

    # Generate submission
    test_output = trainer.predict(torch.FloatTensor(X_test_np)).cpu().data.numpy()
    submission = pd.DataFrame(data=test_output,index=X_test.index, columns=['Next_Premium'])
    
    train_output = trainer.predict(torch.FloatTensor(X_train_np)).cpu().data.numpy()
    submission_train = pd.DataFrame(data=train_output,index=X_train.index, columns=['Next_Premium'])
    
    valid_output = trainer.predict(torch.FloatTensor(X_valid_np)).cpu().data.numpy()
    submission_valid = pd.DataFrame(data=valid_output,index=X_valid.index, columns=['Next_Premium'])

    return {
        'model': trainer, 'submission': submission, 
        'submission_train':submission_train, 'submission_valid':submission_valid,
        'valid_loss':valid_loss, 'summary':summary, 'scaler':scaler
    }


def read_data(file_name, index_col='Policy_Number', path='interim'):
    '''
    In: file_name
    Out: interim_data
    Description: read data from directory /data/interim
    '''
    # set the path of raw data
    interim_data_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), os.path.pardir, os.path.pardir, 'data', path
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

def gen_prem_60(df_policy, save=False):
    # df_policy = read_data('policy_0702.csv', path='raw')

    # remove terminated cols
    real_ia = df_policy['Insured_Amount1'] + df_policy['Insured_Amount2'] + df_policy['Insured_Amount3']

    # rows: policy number; cols: insurance coverage
    prem60 = df_policy[real_ia != 0].set_index('Insurance_Coverage', append=True)[['Premium']].unstack(level=1).fillna(0)
    prem60.columns = [col[1] for col in prem60.columns]

    if save:
        interim_data_path = os.path.join(os.path.dirname('__file__'), os.path.pardir, os.path.pardir, 'data', 'interim')
        write_sample_path = os.path.join(interim_data_path, 'premium_60.csv')
        prem60.to_csv(write_sample_path)

    return prem60

def gen_ia_60(df_policy, save=False):

    # rows: policy number; cols: insurance coverage
    ia60 = df_policy.set_index('Insurance_Coverage', append=True)[
        ['Insured_Amount1','Insured_Amount2','Insured_Amount3']
    ].unstack(level=1).fillna(0)
    ia60.columns = ['_'.join(col) for col in ia60.columns]

    if save:
        interim_data_path = os.path.join(os.path.dirname('__file__'), os.path.pardir, os.path.pardir, 'data', 'interim')
        write_sample_path = os.path.join(interim_data_path, 'ia_60.csv')
        ia60.to_csv(write_sample_path)

    return ia60

def gen_cd_60(df_policy, save=False):

    # rows: policy number; cols: insurance coverage
    cd60 = df_policy.set_index('Insurance_Coverage', append=True)[
        ['Coverage_Deductible_if_applied']
    ].unstack(level=1).fillna(0)
    cd60.columns = ['_'.join(col) for col in cd60.columns]

    if save:
        interim_data_path = os.path.join(os.path.dirname('__file__'), os.path.pardir, os.path.pardir, 'data', 'interim')
        write_sample_path = os.path.join(interim_data_path, 'cd_60.csv')
        cd60.to_csv(write_sample_path)

    return cd60

def demo(
    epochs=80, base_lr=0.001, momentum=0.9, weight_decay=0, 
    batch_size=128, optimizer='sgd', dropout=False, seed=random.randint(0,1000), 
    get_test=False, save=False, load=False, reduction=10, raw='prem'
):
    rand_reset(seed)
    df_policy = read_data('policy_0702.csv', path='raw')
    # X = read_data('premium_60.csv', path='interim')
    if raw=='prem':
        X = gen_prem_60(df_policy, save=False)
    elif raw=='ia':
        X = gen_ia_60(df_policy, save=False)
    elif raw=='cd':
        X = gen_cd_60(df_policy, save=True)

    X_test = read_data('X_test_bs.csv', path='interim')

    y_train = read_data('y_train_prefs.csv', path='interim')
    y_valid = read_data('y_valid_prefs.csv', path='interim')

    X_train = X.loc[y_train.index]
    X_valid = X.loc[y_valid.index]
    X_test = X.loc[X_test.index]

    print('X_train size: {}'.format(len(X_train)))
    print('y_train size: {}'.format(len(y_train)))
    print('X_valid size: {}'.format(len(X_valid)))
    print('y_valid size: {}'.format(len(y_valid)))
    print('test size: {}'.format(len(X_test)))

    feature_list = [feature for feature in X_train.columns.values if 'cat_' not in feature]
    num_features = len(feature_list)
    print('Number of features: {}'.format(num_features))

    # Filter features
    X_train = X_train[feature_list]
    X_valid = X_valid[feature_list]
    X_test = X_test[feature_list]
    # print(X_valid[0:100])

    ### Fill Missing Values
    X_train = X_train.apply(lambda x:x.fillna(0))
    X_valid = X_valid.apply(lambda x:x.fillna(0))
    X_test = X_test.apply(lambda x:x.fillna(0))

    # begin training
    num_neuron = [100,reduction]
    # num_neuron = [round(1.5*num_features),round(0.3*num_features),round(0.1*num_features)]
    # num_neuron = [160,30,8]
    print('Network Architecture: {}'.format(num_neuron))

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
    
    model_output = get_submission(
        X_train, X_valid, y_train, y_valid, X_test, 
        model=MLPRegressor, max_epoch=epochs, base_lr=base_lr, 
        momentum=momentum, weight_decay=weight_decay,
        batch_size = batch_size, train_params=train_params, 
        test_along=True, optimizer=optimizer, hyper=optim_hyper,
        save=False, load=load, mdl_name='{}60_{}.pt'.format(raw,reduction)
    )

    summary = model_output['summary']
    summary += '>>> random seed: {}\n'.format(seed)

    print(summary)

    # generate submission
    if save:
        # # remove terminated cols
        real_ia = df_policy['Insured_Amount1'] + df_policy['Insured_Amount2'] + df_policy['Insured_Amount3']
        real_cd = df_policy['Coverage_Deductible_if_applied']
        # df_policy = df_policy[real_ia != 0]

        scaler = model_output['scaler']

        # transform dataframe to matrix
        if raw=='prem':
            df_policy_iapos = df_policy.assign(Premium = np.where(np.logical_and(real_ia!=0, real_cd>=0), df_policy['Premium'], 0))
            mtx_df = df_policy_iapos.set_index('Insurance_Coverage', append=True)[['Premium']].unstack(level=1).fillna(0)
        elif raw=='ia':
            mtx_df = df_policy.set_index('Insurance_Coverage', append=True)[
                ['Insured_Amount1','Insured_Amount2','Insured_Amount3']
            ].unstack(level=1).fillna(0)
        elif raw=='cd':
            mtx_df = df_policy.set_index('Insurance_Coverage', append=True)[
                ['Coverage_Deductible_if_applied']
            ].unstack(level=1).fillna(-2)

        # nn dimension reduction
        model = model_output['model'].model
        model.eval() # evaluation mode
        inp = torch.FloatTensor(scaler.transform(mtx_df.values))
        with torch.no_grad():
            if torch.cuda.is_available():
                inp = torch.autograd.Variable(inp.cuda())
            else:
                inp = torch.autograd.Variable(inp)
        nn_df = inp
        modulelist = list(model.regressor.modules())
        for l in modulelist[1:-2]:
        # for l in modulelist[1:]:
            nn_df = l(nn_df)
        nn_df = nn_df.cpu().data.numpy()
        non_cons_cols = np.var(nn_df, axis=0) != 0
        nn_df = nn_df[:,non_cons_cols]
        print('>>> constant columns: {}'.format(len(non_cons_cols)))

        n_comp = nn_df.shape[1]
        print('>>> number of reduced features: {}'.format(n_comp))
        real_ic_nn = pd.DataFrame(nn_df, index = mtx_df.index).fillna(0)
        real_ic_nn.columns = ['real_{}_ic_nn_'.format(raw) + str(i) for i in range(1, n_comp+1)]

        interim_data_path = os.path.join(
            os.path.dirname('__file__'), os.path.pardir, os.path.pardir, 'data', 'interim'
        )
        write_sample_path = os.path.join(interim_data_path, '{}_60_{}.csv'.format(raw,n_comp))
        real_ic_nn.to_csv(write_sample_path)


def rand_reset(seed):
    random.seed(seed)
    torch.manual_seed(random.randint(0,1000))
    torch.cuda.manual_seed_all(random.randint(0,1000))
    np.random.seed(random.randint(0,1000))

if __name__ == '__main__':
    # Example usage: "python nn_train.py --epochs 100"
    fire.Fire(demo)