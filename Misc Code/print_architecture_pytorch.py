# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 18:40:05 2021

@author: jyotm
"""
from torchinfo import summary
from torch import nn as nn
import torch
input_channels, output_channels, multiply_rate = 1, 4, 2
num_points = 2024
conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=(3,3)),
            nn.MaxPool2d(2,2),
            nn.Conv2d(output_channels, output_channels * multiply_rate, kernel_size = (3,3)),
            nn.MaxPool2d(2,2),
            nn.LeakyReLU(),
            nn.Conv2d(output_channels * multiply_rate, output_channels * (multiply_rate**2), kernel_size = (5,5)),
            nn.MaxPool2d(2,2),
            nn.Conv2d(output_channels * (multiply_rate**2), output_channels * (multiply_rate**3), kernel_size = (5,5)),
            nn.MaxPool2d(2,2),
            nn.LeakyReLU(),
            nn.Conv2d(output_channels * (multiply_rate**3), output_channels * (multiply_rate**3), kernel_size = (7,7)),
            nn.MaxPool2d(2,2),
            nn.Conv2d(output_channels * (multiply_rate**3), output_channels * 10, kernel_size = (7,7)),
            nn.MaxPool2d(2,2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(num_features = output_channels * 10),
            nn.Dropout(p=0.2),
            nn.Flatten(start_dim = 2,end_dim = -1),
            nn.Linear(676, 252),
            nn.Flatten(),
            nn.Linear(10080, num_points)
            )
print(summary(conv1, input_size=(1, 2048, 2048)))
torch.cuda.empty_cache()