
from numpy import asarray
from PIL import Image
import numpy as np
import os
import torch
from model import Net
import glob
import pandas as pd
import numpy as np

device = 'cpu'

model = Net().to(device)
model = model.eval()
model.load_state_dict(                                                           
         torch.load(                                                          
              '../outputs/model.pth',map_location=device)['model_state_dict'])

image=Image.open("/Users/lihongfei/Desktop/CS231n/input/GTSRB_Final_Test_Images/GTSRB/Final_Test/images/00000.ppm")
# resize image
newsize = (32, 32)
rimage = image.resize(newsize)
# convert image in form of numpy array
numpydata = asarray(rimage)
#print(numpydata)
# normalize data
nordata = np.zeros((32,32,3))
meann = [0.485, 0.456, 0.406]
stdd = [0.229, 0.224, 0.225]
for i in range(3):
   cmax = np.max(numpydata[:,:,i])
   #nordata[:,:,i] = (numpydata[:,:,i] - meann[i] * cmax) / (stdd[i] * cmax)
   nordata[:,:,i] = (numpydata[:,:,i] - meann[i]*255) / (stdd[i]*255)
nordata = nordata.astype('float32')
print(nordata.shape)
nordata = np.transpose(nordata, (2, 0, 1)) # default: reverse order
print(nordata.shape)
nordata = nordata.reshape((1,3,32,32))
nordata = torch.from_numpy(nordata)
print(nordata)
output = model(nordata)
noo, preds = torch.max(output, 1)
print(preds)
