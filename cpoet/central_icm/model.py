# Derived from https://github.com/chagmgang/pytorch_ppo_rl/blob/master/model.py
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import math
from torch.nn import init
from gymnasium.spaces import Box, Discrete

class ICMModel(nn.Module):
    def __init__(self, input_size, feature_size, hidden_size, output_size, action_space=None):
        super(ICMModel, self).__init__()

        if type(input_size) == int or len(input_size) == 1:
            self.image=False
            self.input_size = input_size if type(input_size) == int else input_size[0]
        else:
            self.image=True
            self.input_size = input_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.output_size = output_size       
        self.action_space = action_space

        if self.image:
            self.feature = nn.Sequential(
                nn.Conv2d(
                    in_channels=input_size[2],
                    out_channels=32,
                    kernel_size=8,
                    stride=4),
                nn.LeakyReLU(),
                nn.BatchNorm2d(num_features=32),
                nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=4,
                    stride=2),
                nn.LeakyReLU(),
                nn.BatchNorm2d(num_features=64),
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=3,
                    stride=1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(num_features=64),
                nn.Flatten(start_dim=1),
                nn.Linear(
                    7 * 7 * 64,
                    self.feature_size),
                nn.LeakyReLU(),
                nn.BatchNorm1d(num_features=self.feature_size),
            )
        else:
            self.feature = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                nn.LeakyReLU(),
                nn.BatchNorm1d(num_features=self.hidden_size),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LeakyReLU(),
                nn.BatchNorm1d(num_features=self.hidden_size),
                nn.Linear(self.hidden_size, self.feature_size),
                nn.BatchNorm1d(num_features=self.feature_size),
                # large scale study didn't use an activation here 
            )

        self.inverse_model = nn.Sequential(
            nn.Linear(self.feature_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, output_size),
            # large scale study didn't use an activation on features, inverse, or forward 
            # https://github.com/openai/large-scale-curiosity/blob/master/auxiliary_tasks.py
        )

        self.forward_model = nn.Sequential(
            nn.Linear(output_size + self.feature_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.feature_size),
            # nn.Tanh(),

        )

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.kaiming_uniform_(p.weight, a=1.0)
                p.bias.data.zero_()

    def forward(self, inputs, train=True):
        state, next_state, action = inputs
        if self.image:
            state = torch.permute(state, (0, 3, 1, 2)).float()
            next_state = torch.permute(next_state, (0, 3, 1, 2)).float()
        if isinstance(self.action_space, Discrete):    
            if len(action.shape)>1:
                action = action.squeeze()
            action = torch.nn.functional.one_hot(action, self.action_space.n).float()
        encode_state = self.feature(state)
        encode_next_state = self.feature(next_state)
        if train:
            pred_action = torch.cat((encode_state, encode_next_state), 1) 
            pred_action = self.inverse_model(pred_action)
        else:
            pred_action = None

        # ---------------------
            
        # get pred next state
        pred_next_state_feature = torch.cat((encode_state, action), 1)
        pred_next_state_feature = self.forward_model(pred_next_state_feature)

        real_next_state_feature = encode_next_state
        return real_next_state_feature, pred_next_state_feature, pred_action