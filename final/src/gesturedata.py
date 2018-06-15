import os
import numpy as np
import transformations
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



import random
import json,cv2
import show



# Dataset class for memory efficient data processing 
class HandGestureDataset(torch.utils.data.Dataset):
    """ Hand Gesture Dataset """

    def __init__(self, root_dir, transform=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.joints = json.load(open(self.root_dir+"scaled_annotations.json","r"))
        self.names = list(self.joints)
        self.norm = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.ToPILImage()
        ])
        self.trans = [transformations.flip_h, transformations.flip_v,transformations.rotate]
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        idx = np.random.randint(0, len(self.names))
        name = self.names[idx]
        image = Image.open("Dataset/ColoredProcessedImages/" + self.names[idx] + ".png")
        temp = []
        for i in range(len(self.joints[name])):
            temp.append([int(self.joints[name][i][0]), int(self.joints[name][i][1])])
        s_image, s_temp = transformations.resize(image, temp)
        n_image, n_temp = self.trans[random.randrange(0, len(self.trans))](s_image,s_temp)
        orig_image = np.array(n_image)
        n_image = self.norm(n_image)
        #n_image, n_temp = self.trans[random.randrange(0, len(self.trans))](image,temp)
        r_image = np.array(n_image)
        temp = []
            
        sample = {'image': r_image, 
                  #'joints': temp,
                  'joints': n_temp,
                  'name' : name
                 }

        if self.transform:
            sample = self.hog(sample)

        return name, r_image, np.array(n_temp), orig_image
    
    def substract_mean(self, sample):
        sample['image'] = sample['image'].astype(np.float64) - np.mean(sample['image']).astype(np.float64)
        return sample
    
    def hog(self, sample):
        
        sample['image'] = cv2.Laplacian(sample['image'], cv2.CV_64F)
#         sample['image'] = cv2.Sobel(sample['image'], cv2.CV_64F, 1, 0, ksize=5)
#         sample['image'] = cv2.Sobel(sample['image'], cv2.CV_64F, 0, 1, ksize=5)
        return sample