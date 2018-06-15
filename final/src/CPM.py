from torch.autograd import Variable
from torch.utils.data import sampler
import torch.nn.functional as F
from torchvision import models
import torch.nn as nn


class ConvolutionalPoseEstimator(nn.Module):
    def __init__(self, N_keypoints):
        super(ConvolutionalPoseEstimator, self).__init__()
          
        self.N_keypoints = N_keypoints    
        # Stage 1 
        self.conv1_1 = nn.Conv2d(3, 128, kernel_size=9, padding=4)
        self.pool1_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_1 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.pool2_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3_1 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.pool3_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv4_1 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.conv5_1 = nn.Conv2d(32, 512, kernel_size=9, padding=4)
        self.conv6_1 = nn.Conv2d(512, 512, kernel_size=1)
        self.conv7_1 = nn.Conv2d(512, N_keypoints + 1, kernel_size=1)
        self.pool_center = nn.AvgPool2d(kernel_size=9, stride=8, padding=1)
        
        self.conv1_2 = nn.Conv2d(3, 128, kernel_size=9, padding=4)
        self.pool1_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.pool2_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.pool3_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv4_2 = nn.Conv2d(128, 32, kernel_size=5, padding=2)

        self.Mconv1_2 = nn.Conv2d(32 + N_keypoints + 2, 128, kernel_size=11, padding=5)
        self.Mconv2_2 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv3_2 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv4_2 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mconv5_2 = nn.Conv2d(128, N_keypoints + 1, kernel_size=1, padding=0)

    def forward(self):
        pass