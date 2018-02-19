import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
import cv2

import torch


from getfeatures import getfeatures

import torch.nn as nn
from torchvision import models
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F

a = getfeatures()


img1 = cv2.imread('i2.png')
img2 = cv2.imread('i3.png')

b1 = a.extract(img1)
maxfilt1 = b1[0]
# maxfilt1 = np.amax((np.amax(b, axis = 0)),axis=0)
# print(b.shape)
# print((maxfilt1.shape))
b2 = a.extract(img2)
maxfilt1 = b2[0]
# maxfilt2 = np.amax((np.amax(b, axis = 0)),axis=0)

img = img1

reshapedimage = img
reshapedimage = cv2.resize(img,(299, 299), interpolation = cv2.INTER_CUBIC)
transform = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([ transforms.ToTensor(),normalize])
transformedimage = preprocess(reshapedimage)
image_variable = Variable(transformedimage)
image_variable = image_variable.float()

image_variable = image_variable.unsqueeze(0)

img_input = torch.autograd.Variable(image_variable.data.cpu() , requires_grad = True)
img_output = a.inceptionfeaturesmodel(img_input)
img_output.backward(gradient = torch.ones(img_output.size()), retain_variables = True)
# img_output.backward(gradient=torch.ones(img_input.size(), retain_variables=True)
