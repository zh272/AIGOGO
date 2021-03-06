import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class MLPRegressor(nn.Module):
    def __init__(self, num_input, num_neuron, dropout=False):
        super().__init__()
        
        self.regressor = nn.Sequential(OrderedDict([
            ('fc0', nn.Linear(num_input, num_neuron[0])),
            ('fc0_relu', nn.ReLU(inplace=True))
        ]))
        if dropout:
            self.regressor.add_module('fc0_dropout', nn.Dropout())
        for idx in range(1,len(num_neuron)):
            self.regressor.add_module('fc{}'.format(idx), nn.Linear(num_neuron[idx-1], num_neuron[idx]))
            self.regressor.add_module('fc{}_relu'.format(idx), nn.ReLU(inplace=True))
            # self.regressor.add_module('fc{}_dropout'.format(idx), nn.Dropout())
        self.regressor.add_module('fc{}'.format(len(num_neuron)), nn.Linear(num_neuron[-1],1))
        self.regressor.add_module('fc{}_relu'.format(len(num_neuron)), nn.ReLU(inplace=True))

        # Initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                n = param.numel()
                param.data.normal_().mul_(math.sqrt(2. / n))
            elif 'bias' in name:
                param.data.fill_(0)
        
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.regressor(x)


class MLPClassifier(nn.Module):
    def __init__(self, num_input, num_neuron, num_class, dropout=False):
        super().__init__()
        
        self.classifier = nn.Sequential(OrderedDict([
            ('fc0', nn.Linear(num_input, num_neuron[0])),
            ('fc0_relu', nn.ReLU(inplace=True))
        ]))
        if dropout:
            self.classifier.add_module('fc0_dropout', nn.Dropout())
        for idx in range(1,len(num_neuron)):
            self.classifier.add_module('fc{}'.format(idx), nn.Linear(num_neuron[idx-1], num_neuron[idx]))
            self.classifier.add_module('fc{}_relu'.format(idx), nn.ReLU(inplace=True))
            # self.classifier.add_module('fc{}_dropout'.format(idx), nn.Dropout())
        self.classifier.add_module('fc{}'.format(len(num_neuron)), nn.Linear(num_neuron[-1],num_class))

        # Initialization
        # for name, param in self.named_parameters():
        #     if 'weight' in name:
        #         n = param.numel()
        #         param.data.normal_().mul_(math.sqrt(2. / n))
        #     elif 'bias' in name:
        #         param.data.fill_(0)
        
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class ConvNet1D(nn.Module):
    def __init__(self, num_cv_filter=[1,20,40], num_fc_neuron = [20,10], dropout=True):
        super().__init__()
        self.features = nn.Sequential()
        for idx in range(len(num_cv_filter)-1):
            num_in = num_cv_filter[idx]
            num_out = num_cv_filter[idx+1]

            self.features.add_module(
                'conv{}'.format(idx),  nn.Conv1d(num_in, num_out, kernel_size=5, stride=1, padding=0)
            )
            # if dropout:
            #     self.features.add_module('conv{}_drop'.format(idx), nn.Dropout())

            # self.features.add_module('conv{}_avgpool'.format(idx), nn.AvgPool1d(kernel_size=2, stride=2, padding=0))
            self.features.add_module('conv{}_maxpool'.format(idx), nn.MaxPool1d(kernel_size=2, stride=2, padding=0))
            self.features.add_module('conv{}_relu'.format(idx), nn.ReLU(inplace=True))

        self.regressor = nn.Sequential()
        for idx in range(len(num_fc_neuron)-1):
            if idx == 0:
                num_in = num_cv_filter[-1]
                num_out = num_fc_neuron[0]
            else:
                num_in = num_fc_neuron[idx-1]
                num_out = num_fc_neuron[idx]
            self.regressor.add_module('fc{}'.format(idx), nn.Linear(num_in, num_out))
            self.regressor.add_module('fc{}_relu'.format(idx), nn.ReLU(inplace=True))
            if dropout:
                self.regressor.add_module('fc{}_dropout'.format(idx), nn.Dropout())
        # last fc layer
        self.regressor.add_module('fc{}'.format(len(num_fc_neuron)-1), nn.Linear(num_fc_neuron[-2],num_fc_neuron[-1]))

    def forward(self, x):
        features = self.features(x.view(x.shape[0], 1, x.shape[1]))

        # features = F.avg_pool1d(features, kernel_size=features.size(-1))
        features = F.max_pool1d(features, kernel_size=features.size(-1))

        features = features.view(features.size(0), -1)
        out = self.regressor(features)

        # return F.log_softmax(out, dim=1)
        return out