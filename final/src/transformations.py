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



def convert_coordinates(origin, point):
    new_x = point[0] - origin[0]
    new_y = origin[1] - point[1]
    return (new_x, new_y)

def calculate_origin(width, height):
    new_width = width//2
    new_height = height//2
    return (new_width, new_height)

def revert_coordinates(origin, point):
    orig_x = point[0] + origin[0]
    orig_y = origin[1] - point[1]
    return (orig_x, orig_y)

def reflect_across_y(point, width):
    array_x = width - 1
    old_x = point[0]
    old_y = point[1]
    return (array_x-old_x, old_y)

def reflect_across_x(point, height):
    array_y = height - 1
    old_x = point[0]
    old_y = point[1]
    return (old_x, array_y - old_y)

def flip_h(img, joints):
    n_image = T.functional.hflip(img)
    new_joints = []
    for i in joints:
        point = (i[0], i[1])
        r_point = reflect_across_y(point, img.size[0])
        new_joints.append([r_point[0], r_point[1]])
    return n_image, new_joints

def flip_v(img, joints):
    n_image = T.functional.vflip(img)
    new_joints = []
    for i in joints:
        point = (i[0], i[1])
        r_point = reflect_across_x(point, img.size[1])
        new_joints.append([r_point[0], r_point[1]])
    return n_image, new_joints

def zoom_in(img, joints):
    pass

def resize(img, joints):
    new_size = random.randint(100, 200)
    n_image = T.functional.resize(img, 224)
    new_width = n_image.size[0]
    new_height = n_image.size[1]
    p_width = new_width/img.size[0]
    p_height = new_height/img.size[1]
    new_joints = []
    for i in joints:
        new_x = int(p_width * i[0])
        new_y = int(p_height * i[1])
        new_joints.append([new_x, new_y])
    return n_image, new_joints

def rotate(img, joints):
    origin = calculate_origin(img.size[0],img.size[1])
    
    n_image = T.functional.rotate(img, 45)
    new_joints = []
    for i in joints:
        point = (i[0],i[1])
        converted_point = convert_coordinates(origin, point)
        radian = math.radians(45)
        new_x = converted_point[0] * math.cos(radian) - converted_point[1] * math.sin(radian)
        new_y = converted_point[1] * math.cos(radian) + converted_point[0] * math.sin(radian)
        
        new_point = (int(new_x),int(new_y))
        final_point = revert_coordinates(origin, new_point)
        new_joints.append([final_point[0], final_point[1]])
        
    return n_image, new_joints