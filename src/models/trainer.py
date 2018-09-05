import torch
import random
import numpy as np
from copy import deepcopy
from torch.autograd import Variable

from helpers import AverageMeter, get_optimizer, test_epoch, WeightedSubsetRandomSampler


class Trainer:
    def __init__(self, model, train_set, loss_fn, valid_set=None, weights=None, hyper={}, 
                batch_size=64, valid_size=0, epochs=None, optimizer='sgd'):
        if torch.cuda.is_available():
            # Wrap model for multi-GPUs, if necessary
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(model).cuda()
            else:
                self.model = model.cuda()
        else:
            self.model = model
        self.optimizer, self.scheduler, self.hyper = get_optimizer(
            model=model, hyper=hyper, epochs=epochs, optimizer=optimizer
        )
        self.evaluator_loss = AverageMeter()
        self.evaluator_err = AverageMeter()
        self.num_params = 0
        self.layer_idx = [0]
        for p in self.model.parameters():
            if p.requires_grad:
                self.num_params += p.numel()
                self.layer_idx.append(self.num_params)
        self.loss_fn = loss_fn
        self.batch_size = batch_size

        # Create train/valid split
        self.num_train = len(train_set)
        if valid_set is None:
            self.num_valid = int(round(valid_size*self.num_train))
        else:
            self.num_valid = len(valid_set)
        self.train_set = train_set
        self.valid_set = valid_set
        self.weights = weights
        self.reset_train_valid()

    def reset_train_valid(self):
        if self.valid_set is None:
            indices = torch.randperm(self.num_train)
            train_indices = indices[:self.num_train - self.num_valid]
            valid_indices = indices[self.num_train - self.num_valid:]
            if self.weights is None:
                # train_sampler = WeightedSubsetRandomSampler(train_indices,None)
                train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
            else:
                train_sampler = WeightedSubsetRandomSampler(train_indices, self.weights[train_indices])
            valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)



            # Data loaders
            self.train_loader = torch.utils.data.DataLoader(
                self.train_set, batch_size=self.batch_size, sampler=train_sampler,
                pin_memory=(torch.cuda.is_available()), num_workers=0, drop_last=False
            )
            self.valid_loader = torch.utils.data.DataLoader(
                self.train_set, batch_size=self.batch_size, sampler=valid_sampler,
                pin_memory=(torch.cuda.is_available()), num_workers=0, drop_last=False
            )

        else:
            train_indices = torch.randperm(self.num_train)
            valid_indices = torch.randperm(self.num_valid)
            if self.weights is None:
                # train_sampler = WeightedSubsetRandomSampler(train_indices,None)
                train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
            else:
                train_sampler = WeightedSubsetRandomSampler(train_indices, self.weights[train_indices])
            valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)

            # Data loaders
            self.train_loader = torch.utils.data.DataLoader(
                self.train_set, batch_size=self.batch_size, sampler=train_sampler,
                pin_memory=(torch.cuda.is_available()), num_workers=0, drop_last=False
            )
            self.valid_loader = torch.utils.data.DataLoader(
                self.valid_set, batch_size=self.batch_size, sampler=valid_sampler,
                pin_memory=(torch.cuda.is_available()), num_workers=0, drop_last=False
            )

        self.train_loader_iter = iter(self.train_loader)
        self.valid_loader_iter = iter(self.valid_loader)
        self.train_loader_list = list(self.train_loader)
        self.valid_loader_list = list(self.valid_loader)
        self.train_batch = len(self.train_loader)
        self.valid_batch = len(self.valid_loader)

    def train_epoch(self):
        if self.scheduler:
            self.scheduler.step()
        for _, (inputs, target) in enumerate(self.train_loader):
            self.step(inputs, target)

    def step(self, inputs, target):
        """Forward pass and backpropagation"""
        self.model.train() # enter train mode
        if torch.cuda.is_available():
            inputs = Variable(inputs.cuda())
            target = Variable(target.cuda())
        else:
            inputs = Variable(inputs)
            target = Variable(target)

        output = self.model(inputs)
        loss = self.loss_fn(output, target)
        # loss = self.loss_fn(output, target.view_as(output))
        # loss = self.loss_fn(output.view_as(target), target)
        if np.isnan(loss.item()):
            # print(output)
            # print(target)
            print('WARNING at trainer.py: Loss is NaN!')

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, inp):
        """Evaluate model on the provided validation or test set."""
        self.model.eval() # evaluation mode

        with torch.no_grad():
            if torch.cuda.is_available():
                inp = Variable(inp.cuda())
            else:
                inp = Variable(inp)
        
        output = self.model(inp)
        # pred = output.data.max(1, keepdim=True)[1]
        return output

    def eval(self, load='train', batch_id=None, eval_type='loss'):
        """Evaluate model on the provided validation or test set."""
        self.model.eval() # evaluation mode
        if batch_id is None:
            loader = self.valid_loader_iter if load=='valid' else self.train_loader_iter
            inp, target = next(loader, (None,None))
        else:
            loader = self.valid_loader_list if load=='valid' else self.train_loader_list
            inp, target =loader[batch_id]

        if target is None:
            if load=='valid':
                self.valid_loader_iter = iter(self.valid_loader)
                inp, target = next(self.valid_loader_iter)
            else:
                self.train_loader_iter = iter(self.train_loader)
                inp, target = next(self.train_loader_iter)

        with torch.no_grad():
            if torch.cuda.is_available():
                # inp = Variable(inp.cuda())
                target = Variable(target.cuda())
            else:
                # inp = Variable(inp)
                target = Variable(target)
        
        output = self.predict(inp)

        
        if eval_type=='loss':
            result = self.loss_fn(output, target).item()
            self.evaluator_loss.update(result, len(target))
        else:
            pred = output.data.max(1, keepdim=True)[1]
            incorrect = pred.ne(target.data.view_as(pred)).cpu().sum()
            result = incorrect.item() / len(target)

            self.evaluator_err.update(result, len(target))
        # return self.evaluator.avg
        return result
    
    def loss(self, load='train', batch_id=None):
        self.model.eval()
        if batch_id is None:
            loader = self.valid_loader_iter if load=='valid' else self.train_loader_iter
            inp, target = next(loader, (None,None))
        else:
            loader = self.valid_loader_list if load=='valid' else self.train_loader_list
            inp, target =loader[batch_id]

        if target is None:
            if load=='valid':
                self.valid_loader_iter = iter(self.valid_loader)
                inp, target = next(self.valid_loader_iter)
            else:
                self.train_loader_iter = iter(self.train_loader)
                inp, target = next(self.train_loader_iter)

        with torch.no_grad():
            if torch.cuda.is_available():
                inp = Variable(inp.cuda())
                target = Variable(target.cuda())
            else:
                inp = Variable(inp)
                target = Variable(target)

        output = self.model(inp)
        loss = self.loss_fn(output, target)
        return loss.item()

    def loss_epoch(self, load='valid', eval_type='loss'):
        loader = self.valid_loader if load=='valid' else self.train_loader
        loss_avg, error_avg = test_epoch(self.model, loader, loss_fn=self.loss_fn, eval_type=eval_type)
        if eval_type=='loss':
            return loss_avg
        else:
            return error_avg

    def reset_evaluator(self):
        self.evaluator.reset()

