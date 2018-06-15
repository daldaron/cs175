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


import timeit, random
import json,cv2
import show












class PoseEstimator(nn.Module):
    def __init__(self, pretrained_conv):
        super(PoseEstimator, self).__init__()
        for param in pretrained_conv.parameters():
            param.requires_grad = False
        
#         modules = list(pretrained_conv.children())     # delete the last fc layer.
#         self.conv = nn.Sequential(*modules)
        self.conv = pretrained_conv
        self.fc1 = nn.Linear(1000, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 42)
    
    def forward(self, x):
        x = self.conv(x)
        #N, C, H, W = x.size()
        #x =  x.view(N, -1)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.relu(x)

        return x
    