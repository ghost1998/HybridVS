import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
import cv2

import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F
import time
from inceptionfeatures import inceptionfeatures
from photometricvsutils import getfeatures
from photometricvsutils import getL3



# ---------------------------------------------------------------------------------------------------------------------
inception = models.inception_v3(pretrained=True)
inceptionfeaturesmodel = inceptionfeatures(inception)

std=[0.229, 0.224, 0.225]

transform = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([
    transforms.ToTensor(),
    normalize
    ])
border = 5

Z = 4

# bord = 5
camK =  [
            [205.4696 , 0 ,  320.5000],
            [0  , 205.4696 , 240.5000],
            [0, 0, 1]
        ]
camK = np.asarray(camK)


# ---------------------------------------------------------------------------------------------------------------------
img = cv2.imread('i1.png')
imgd = img
reshapedimage = img
reshapedimage = cv2.resize(img,(299, 299), interpolation = cv2.INTER_CUBIC)
transformedimage = preprocess(reshapedimage)
transformedimageinput = transformedimage.unsqueeze(0)
image_variable = Variable(transformedimageinput, requires_grad=True)
prediction = inceptionfeaturesmodel(image_variable)

features_des = getfeatures(prediction[0].cpu().data.numpy())
img_normalized_des = transformedimage.cpu().numpy()

img = cv2.imread('i2.png')
reshapedimage = img
reshapedimage = cv2.resize(img,(299, 299), interpolation = cv2.INTER_CUBIC)
transformedimage = preprocess(reshapedimage)
transformedimageinput = transformedimage.unsqueeze(0)
image_variable = Variable(transformedimageinput, requires_grad=True)
prediction = inceptionfeaturesmodel(image_variable)

features = getfeatures(prediction[0].cpu().data.numpy())
img_normalized = transformedimage.cpu().numpy()


error = (features - features_des)


# ---------------------------------------------------------------------------------------------------------------------
k = 0
L1 = np.zeros((prediction[0].size()[0] * prediction[0].size()[1] , 3*(reshapedimage.shape[0]-2 *border) *(reshapedimage.shape[1]-2 *border)))
st_time = time.time()
for i in range(prediction[0].size()[0]):
    print(i)
    if(i>3):
        break
    for j in range(prediction[0].size()[1]):
        prediction[0][i][j].backward(retain_variables=True)
        gradatpoint = image_variable.grad.data.cpu().numpy()
        gradatpoint_withborders =  gradatpoint[0, :,  border : reshapedimage.shape[0]-border, border: reshapedimage.shape[1]-border]
        gradatpoint_flat = np.copy(gradatpoint_withborders.flatten())
        L1[k] = gradatpoint_flat
        image_variable.grad.data.zero_()
        k = k+1
print(time.time() - st_time)





# ---------------------------------------------------------------------------------------------------------------------
# Check this again because in the Harit code, L is caluclate for desired. Ask him
L3 = np.zeros(((3*(reshapedimage.shape[0]-2 *border) *(reshapedimage.shape[1]-2 *border)), 6))
L3[:(reshapedimage.shape[0]-2 *border) *(reshapedimage.shape[1]-2 *border)]  = getL3(img_normalized[0],camK,Z, border )
L3[(reshapedimage.shape[0]-2 *border) *(reshapedimage.shape[1]-2 *border) :2 * (reshapedimage.shape[0]-2 *border) *(reshapedimage.shape[1]-2 *border)] = getL3(img_normalized[1],camK,Z, border )
L3[2*(reshapedimage.shape[0]-2 *border) *(reshapedimage.shape[1]-2 *border) :3 * (reshapedimage.shape[0]-2 *border) *(reshapedimage.shape[1]-2 *border)] = getL3(img_normalized[2],camK,Z, border )



# ---------------------------------------------------------------------------------------------------------------------
L = np.matmul(L1, L3)
