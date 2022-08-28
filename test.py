from numpy import asarray
from PIL import Image
import numpy as np
image =Image.open("/Users/lihongfei/Desktop/CS231n/input/GTSRB_Test/Images/00021.ppm")
# resize image
newsize = (32, 32)
rimage = image.resize(newsize)
# convert image in form of numpy array
numpydata = asarray(rimage)
# normalize data
nordata = np.zeros((32,32,3))
meann = [0.485, 0.456, 0.406]
stdd = [0.229, 0.224, 0.225]

for i in range(3):
    cmax = np.max(np.max(numpydata[:,:,i]))
    nordata[:,:,i] = (numpydata[:,:,i] - meann[i] * cmax) / (stdd[i] * cmax)

#print(nordata)
#print(nordata.shape)
nordata = nordata.astype('float32')
nordata = np.transpose(nordata)
nordata = nordata.reshape((1,3,32,32))
import torch
nordata = torch.from_numpy(nordata)
print(nordata.dtype)
print(nordata.shape)
import torch
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.nn import functional as F
from model import Net
from torch.utils.data import DataLoader
BATCH_SIZE = 1
NUM_WORKERS = 1 

#test_loader = DataLoader(nordata, batch_size=BATCH_SIZE, 
#       shuffle=False, num_workers=NUM_WORKERS)
#
#print(test_loader)

#image, label = test_loader
#image = image.to(device)

device = 'cpu'

model = Net().to(device)
model = model.eval()
model.load_state_dict(
            torch.load(
                        '../outputs/model.pth', map_location=device
                            )['model_state_dict']
            )

output = model(nordata)

noo, preds = torch.max(output, 1)
print(preds)
