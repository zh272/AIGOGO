import os
import torch
import pickle
import numpy as np
import pandas as pd
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
from sklearn import preprocessing
from torch.autograd import Variable
from bisect import bisect_right, bisect_left
from sklearn.model_selection import train_test_split


class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode
        self.le_list = {}

    def fit(self,X,y=None):
        if self.columns is None:
            for colname,col in X.items():
                self.le_list[colname] = preprocessing.LabelEncoder()
                self.le_list[colname].fit(col.astype(str))
        else:
            for colname in self.columns:
                self.le_list[colname] = preprocessing.LabelEncoder()
                self.le_list[colname].fit(X[colname].astype(str))
        return self

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        for colname, le in self.le_list.items():
            try:
                output[colname] = le.transform(output[colname].astype(str))
            except ValueError:
                print('New Labels Found in column [{}]'.format(colname))
                if colname == "('Insurance_Coverage', 'Liability')":
                    print('')
                if colname == "Distribution_Channel":
                    print('')
                le.classes_ = np.append('0000000', le.classes_) # treat all unseen labels as on class
                classes = np.unique(output[colname].astype(str))
                diff = np.setdiff1d(classes, le.classes_)
                for value in diff:
                    # output[colname].iloc[output[colname][output[colname]==value]] = np.nan
                    indices = output[colname][output[colname].astype(str)==value].index.tolist()
                    for idx in indices:
                        output.at[idx, colname] = '0000000'
                output[colname] = le.transform(output[colname].astype(str))
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)



def get_dataset(X_train, y_train, X_test, X_valid=None, y_valid=None, valid_size=None, target_type=torch.FloatTensor, target_shape=None):
    if target_shape:
        y_train = y_train.reshape(*target_shape)
        if y_valid is not None:
            y_valid = y_valid.reshape(*target_shape)
    if valid_size is not None:
        X_train, X_valid, y_train, y_valid = train_test_split(X_train,y_train,test_size=valid_size,random_state=101)
    # scaler = preprocessing.MinMaxScaler()
    scaler = preprocessing.StandardScaler()
    # X_train.data = scaler.fit_transform(X_train)
    X_train = pd.DataFrame(data=scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    if X_valid is not None:
        # X_valid.data = scaler.transform(X_valid)
        X_valid = pd.DataFrame(data=scaler.transform(X_valid), columns=X_valid.columns, index=X_valid.index)
    # X_test.data = scaler.transform(X_test)
    X_test = pd.DataFrame(data=scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    train_set = torch.utils.data.TensorDataset(torch.FloatTensor(X_train.values), target_type(y_train.values))
    if X_valid is None:
        valid_set = None
    else:
        valid_set = torch.utils.data.TensorDataset(torch.FloatTensor(X_valid.values), target_type(y_valid.values))

    return train_set, valid_set, X_test, X_train, y_train, X_valid, y_valid, scaler

def ready(steps, threshold=100, population=None):
    return steps%threshold == 0

def get_optimizer(model, hyper={}, epochs=None, optimizer='sgd'):
    """This is where users choose their optimizer and define the
       hyperparameter space they'd like to search."""
    lr = hyper['lr'] if 'lr' in hyper else np.random.choice(np.logspace(-3, 0, base=10))
    momentum = hyper['momentum'] if 'momentum' in hyper else np.random.choice(np.linspace(0.1, .9999))
    weight_decay = hyper['weight_decay'] if 'weight_decay' in hyper else 0.0001
    hyper['lr'], hyper['momentum'], hyper['weight_decay']= lr, momentum, weight_decay
    if optimizer=='sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=weight_decay)

        if hyper and 'lr_schedule' in hyper:
            scheduler = CustomLR(optimizer, hyper['lr_schedule'])
        elif epochs:
            if 'scheduler' in hyper:
                if hyper['scheduler'] == 'lambdalr':
                    scheduler = optim.lr_scheduler.LambdaLR(
                        optimizer, lr_lambda=lambda epoch: 1/(1+epoch)
                    )
                # elif hyper['scheduler'] == 'plateau':
                #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                #         optimizer, mode='min', verbose=True, patience=10, threshold=0.0001, min_lr=0.0000001
                #     )
            else:
                scheduler = optim.lr_scheduler.MultiStepLR(
                    optimizer,  milestones=[int(0.5 * epochs), int(0.75 * epochs)], gamma=0.1
                )
                hyper['lr_schedule'] = {0:lr, 0.5*epochs:lr*0.1, 0.75*epochs: lr*0.01}
            
            
        else:
            scheduler = None
    elif optimizer=='adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=lr,weight_decay=weight_decay)
        scheduler = None
    elif optimizer=='adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=lr,weight_decay=weight_decay)
        scheduler = None
    elif optimizer=='rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr,momentum=momentum, weight_decay=weight_decay)
        scheduler = None
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
        scheduler = None


    # if hyper and 'lr_schedule' in hyper:
    #     scheduler = CustomLR(optimizer, hyper['lr_schedule'])
    # elif epochs:
    #     scheduler = optim.lr_scheduler.MultiStepLR(
    #         optimizer,  milestones=[int(0.5 * epochs), int(0.75 * epochs)], gamma=0.1
    #     )
    #     hyper['lr_schedule'] = {0:lr, 0.5*epochs:lr*0.1, 0.75*epochs: lr*0.01}
    # else:
    #     scheduler = None

    
        
    return optimizer, scheduler, hyper

class CustomLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, schedule, last_epoch=-1):
        self.milestones = list(schedule.keys())
        self.lrs = list(schedule.values())
        super(CustomLR, self).__init__(optimizer, last_epoch)
    def get_lr(self):
        return [
            self.lrs[
                bisect_right(self.milestones, self.last_epoch)
            ] for _ in self.base_lrs
        ]

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.hist = []
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.hist.append(val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def test_epoch(model, loader, print_freq=1, is_test=True, loss_fn=F.cross_entropy, eval_type='loss'):
    losses = AverageMeter()
    error = AverageMeter()

    # Model on eval mode
    model.eval()

    for _, (inputs, target) in enumerate(loader):
        # Create vaiables
        with torch.no_grad():

            if torch.cuda.is_available():
                input_var = Variable(inputs.cuda(async=True))
                target_var = Variable(target.cuda(async=True))
            else:
                input_var = Variable(inputs)
                target_var = Variable(target)

        # compute output
        output = model(input_var)

        # compute loss
        batch_size = target.size(0)

        if eval_type=='loss':
            loss = loss_fn(output, target_var)
            losses.update(loss.item(), batch_size)
        else:
            # compute prediction error
            _, pred = output.data.cpu().topk(1, dim=1)
            error.update(torch.ne(pred.squeeze().long(), target.cpu().long()).float().sum() / batch_size, batch_size)
        

    # Return summary statistics
    return losses.avg, error.avg

def save_obj(obj, file_name, file_dir='./saved_models'):
    if not os.path.isdir(file_dir): 
        os.makedirs(file_dir)
    
    with open(os.path.join(file_dir,'{}.pkl'.format(file_name)), 'w+b') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(file_name, file_dir='./saved_models'):
    try:
        with open(os.path.join(file_dir,'{}.pkl'.format(file_name)), 'r+b') as f:
            try:
                obj = pickle.load(f)
            except EOFError:
                obj = {}
            return obj
    except FileNotFoundError:
        return {}

def weighted_mae_loss(inp, target, weights=None):
    if weights is None:
        weights = torch.tensor([1.0], device=target.data.device, dtype=target.data.dtype)
    else:
        assert len(weights) == len(target)

    weights = weights.expand_as(target)
    out = (inp-target).abs() * weights
    # expand_as because weights are prob not defined for mini-batch
    loss = out.sum()/weights.sum() # or sum over whatever dimensions
    return loss


class WeightedSubsetRandomSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, indices, weights=None, replacement=True):
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.indices = indices
        if weights is None:
            self.weights = torch.tensor([1.0], dtype=torch.double).expand_as(indices)
        else:
            assert len(weights) == len(indices)
            self.weights = torch.tensor(weights.flatten(), dtype=torch.double)
        self.num_samples = len(self.indices)
        self.replacement = replacement
        
        self.w_i_iter = torch.multinomial(self.weights, self.num_samples, self.replacement)

    def __iter__(self):
        return (self.indices[i] for i in self.w_i_iter)

    def __len__(self):
        return self.num_samples