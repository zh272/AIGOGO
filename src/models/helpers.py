import os
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from bisect import bisect_right, bisect_left
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def get_dataset(X_train, y_train, X_test, target_type=torch.FloatTensor, target_shape=None):
    if target_shape:
        y_train = y_train.reshape(*target_shape)
    # X_train, X_valid, y_train, y_valid = train_test_split(X_train,y_train,test_size=0.2,random_state=101)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_set = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), target_type(y_train))
    # valid_set = torch.utils.data.TensorDataset(torch.FloatTensor(X_valid), target_type(y_valid))

    return train_set, X_test

def ready(steps, threshold=100, population=None):
    return steps%threshold == 0

def get_optimizer(model, hyper={}, epochs=None):
    """This is where users choose their optimizer and define the
       hyperparameter space they'd like to search."""
    lr = hyper['lr'] if 'lr' in hyper else np.random.choice(np.logspace(-3, 0, base=10))
    momentum = hyper['momentum'] if 'momentum' in hyper else np.random.choice(np.linspace(0.1, .9999))
    hyper['lr'], hyper['momentum'] = lr, momentum
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=0.0001)

    if hyper and 'lr_schedule' in hyper:
        scheduler = CustomLR(optimizer, hyper['lr_schedule'])
    elif epochs:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,  milestones=[int(0.5 * epochs), int(0.75 * epochs)], gamma=0.1
        )
        hyper['lr_schedule'] = {0:lr, 0.5*epochs:lr*0.1, 0.75*epochs: lr*0.01}
    else:
        scheduler = None
        
    return optimizer, scheduler, hyper

class CustomLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, schedule, last_epoch=-1):
        self.milestones = list(schedule.keys())
        self.lrs = list(schedule.values())
        super(CustomLR, self).__init__(optimizer, last_epoch)
    def get_lr(self):
        return [
            self.lrs[
                bisect_left(self.milestones, self.last_epoch)
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



def test_epoch(model, loader, print_freq=1, is_test=True, loss_fn=F.cross_entropy):
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
        # loss = loss_fn(output.view_as(target_var), target_var)
        loss = loss_fn(output, target_var)

        batch_size = target.size(0)

        # compute prediction error
        # _, pred = output.data.cpu().topk(1, dim=1)
        # error.update(torch.ne(pred.squeeze().long(), target.cpu().long()).float().sum() / batch_size, batch_size)
        
        losses.update(loss.item(), batch_size)

    # Return summary statistics
    return losses.avg, error.avg
