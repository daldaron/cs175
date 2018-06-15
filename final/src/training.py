import os
import numpy as np

import torch
import torchvision.datasets as dset
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import math
from torch.autograd import Variable
from torch.utils.data import sampler
import torch.nn.functional as F
from skimage import io, transform


import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 16.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
%matplotlib inline

import timeit, random
import json,cv2
import show

%load_ext autoreload
%autoreload 2

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode







def train(model, loss_fn, optimizer, loader_train, loader_val=None, num_epochs=1):
    loss_history = []
    train_acc = []
    val_acc = []
    
    for epoch in range(num_epochs):
        
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        model.train()
        for t, (name, x, y, orig) in enumerate(loader_train):
            x_var = Variable(orig.type(gpu_dtype))
            
            N, C, H, W = x_var.size() # W is C
            x_var =  x_var.view(N, W, C, H)
            y_var = Variable(y.type(gpu_dtype))#.long())
            scores = model(x_var)
            
            
            if t == 0:
                print("scores: {}".format(scores[:10]))
                print("x_var: {}".format(x_var[:10]))
            
            J, K, L = y_var.size()
            y_var = y_var.view(N,-1)
            loss = loss_fn(scores, y_var)
            loss_history.append(float(loss))
            
            if (t + 1) % print_every == 0:
                print('t = %d, loss = %.4f' % (t + 1, loss.data[0]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
#         model.eval()
        
#         for x, y in loader_train:
#             x_var = Variable(x.type(gpu_dtype), volatile=True)

#             scores = model(x_var)
#             _, preds = scores.data.cpu().max(1)
#             num_correct += (preds == y).sum()
#             num_samples += preds.size(0)
#         train_acc.append(float(num_correct) / num_samples)
        
#         for x, y in loader_val:
#             x_var = Variable(x.type(gpu_dtype), volatile=True)

#             scores = model(x_var)
#             _, preds = scores.data.cpu().max(1)
#             num_correct += (preds == y).sum()
#             num_samples += preds.size(0)
#         val_acc.append(float(num_correct) / num_samples)
        
    plt.subplot(1, 1, 1)
    plt.title('Training loss')
    plt.plot(loss_history, 'o', alpha=.05)
    plt.xlabel('Iteration')

#     plt.subplot(2, 1, 2)
#     plt.title('Accuracy')
#     plt.plot(solver.train_acc_history, '-o', label='train')
#     plt.plot(solver.val_acc_history, '-o', label='val')
#     plt.plot([0.7] * len(solver.val_acc_history), 'k--')
#     plt.xlabel('Epoch')
#     plt.legend(loc='lower right')
    plt.gcf().set_size_inches(15, 12)
    plt.show()