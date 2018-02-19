import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
import cv2

import torch


# from getfeatures import getfeatures

import torch.nn as nn
from torchvision import models
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F

from numpy import genfromtxt
import numpy as np
import cv2

from inceptionfeatures import inceptionfeatures




img1 = cv2.imread('test.jpg')
img = img1

# img = np.random.rand(480, 640, 3)

inception = models.inception_v3(pretrained=True)
inceptionfeaturesmodel = inceptionfeatures(inception)
# inceptionfeaturesmodel.cuda()


reshapedimage = img
reshapedimage = cv2.resize(img,(299, 299), interpolation = cv2.INTER_CUBIC)
std=[0.229, 0.224, 0.225]
transform = transforms.ToTensor()
# transformedimage = transform(reshapedimage.astype(float))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
# normalize = transforms.Normalize(mean=[0.456, 0.456, 0.456],std=[0.225, 0.225, 0.225])

# normalize = transforms.Normalize(mean=[0.485, 0.456],std=[0.229, 0.224])
preprocess = transforms.Compose([
    transforms.ToTensor(),
    normalize
    ])

preprocess1 = transforms.Compose([
    transforms.ToTensor(),
    ])

transformedimage = preprocess(reshapedimage)
transformedimageinput = transformedimage.unsqueeze(0)



image_variable = Variable(transformedimageinput, requires_grad=True)
# image_variable = Variable(transformedimageinput.cuda(), requires_grad=True)

prediction = inceptionfeaturesmodel(image_variable)


# prediction.backward(gradient=torch.ones( prediction.size()), retain_variables=True)
prediction.backward(gradient=torch.ones( prediction.size()).cuda(), retain_variables=True)

prediction[0][0][0].backward(retain_variables=True)
a = image_variable.grad
a1d = np.copy(a.data.cpu().numpy())

# inceptionfeaturesmodel.zero_grad()
image_variable.grad.data.zero_()
# prediction.backward(gradient=torch.ones( prediction.size()), retain_variables=True)
prediction[0][18][13].backward(retain_variables=True)
a = image_variable.grad
a5d = np.copy(a.data.cpu().numpy())


print(image_variable.grad)
temp = image_variable.grad
print(temp.size())
# grad_img_input = img_input.grad
if (image_variable.grad is not None):
    print("yes")
else:
    print("no")

# return prediction
k = 0
result = []
L1 = np.zeros((prediction[0].size()[0] * prediction[0].size()[1] , 3*(reshapedimage.shape[0]-2 *border) *(reshapedimage.shape[1]-2 *border)))
st_time = time.time()
for i in range(prediction[0].size()[0]):
    print(i)
    if(i>3):
        break
    for j in range(prediction[0].size()[1]):
        # print(i, j)
        prediction[0][i][j].backward(retain_variables=True)
        gradatpoint = image_variable.grad.data.cpu().numpy()
        # gradatpoint_flat = np.copy(gradatpoint[0].flatten())
        gradatpoint_withborders =  gradatpoint[0, :,  border : reshapedimage.shape[0]-border, border: reshapedimage.shape[1]-border]
        gradatpoint_flat = np.copy(gradatpoint_withborders.flatten())
        L1[k] = gradatpoint_flat
        # result.append(np.copy(gradatpoint[0]))
        image_variable.grad.data.zero_()
        k = k+1
print(time.time() - st_time)

L1 = np.zeros((prediction[0].size()[0] * prediction[0].size()[1] , 3*(reshapedimage.shape[0]-2 *border) *(reshapedimage.shape[1]-2 *border)))
std=[0.229, 0.224, 0.225]

L2_1 = np.ones(3*(reshapedimage.shape[0]-2 *border) *(reshapedimage.shape[1]-2 *border))

L2_1[:(reshapedimage.shape[0]-2 *border) *(reshapedimage.shape[1]-2 *border)] = \
std[0] * L2_1[:(reshapedimage.shape[0]-2 *border) *(reshapedimage.shape[1]-2 *border)]

L2_1[(reshapedimage.shape[0]-2 *border) *(reshapedimage.shape[1]-2 *border) : 2*(reshapedimage.shape[0]-2 *border) *(reshapedimage.shape[1]-2 *border)] = \
std[1] * L2_1[(reshapedimage.shape[0]-2 *border) *(reshapedimage.shape[1]-2 *border) : 2*(reshapedimage.shape[0]-2 *border) *(reshapedimage.shape[1]-2 *border)]

L2_1[2*(reshapedimage.shape[0]-2 *border) *(reshapedimage.shape[1]-2 *border) : 3*(reshapedimage.shape[0]-2 *border) *(reshapedimage.shape[1]-2 *border)] = \
std[2] * L2_1[2*(reshapedimage.shape[0]-2 *border) *(reshapedimage.shape[1]-2 *border) : 3*(reshapedimage.shape[0]-2 *border) *(reshapedimage.shape[1]-2 *border)]


L2 = np.eye(3*(reshapedimage.shape[0]-2 *border) *(reshapedimage.shape[1]-2 *border))
        # if(k>10):
            # break
    # if(k>10):
        # break
