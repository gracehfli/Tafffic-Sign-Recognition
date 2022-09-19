from numpy import asarray
from PIL import Image
import numpy as np
import os
import torch
from model import Net
import glob
import pandas as pd
import numpy as np

path ="/Users/lihongfei/Desktop/CS231n/input/GTSRB_Final_Test_Images/GTSRB/Final_Test/images"
result = []
device = 'cpu'

model = Net().to(device)
model = model.eval()
model.load_state_dict(                                                           
         torch.load(                                                          
              '../outputs/model.pth',map_location=device)['model_state_dict'])
df =  pd.read_csv('/Users/lihongfei/Desktop/CS231n/input/GTSRB_Final_Test_GT/GT-final_test.csv', sep = ';')                                                               
filenames = df['Filename']
gt = np.array(df['ClassId'])
#gt = torch.from_numpy(gt)
path ='/Users/lihongfei/Desktop/CS231n/input/GTSRB_Final_Test_Images/GTSRB/Final_Test/images/'

for i, filename in enumerate(filenames):
    image=Image.open(path+filename)
    # resize image
    newsize = (32, 32)
    rimage = image.resize(newsize)
    # convert image in form of numpy array
    numpydata = asarray(rimage)
    # normalize data
    nordata = np.zeros((32,32,3))
    meann = [0.485, 0.456, 0.406]
    stdd = [0.229, 0.224, 0.225]
    for j in range(3):
        cmax = np.max(np.max(numpydata[:,:,j]))
        nordata[:,:,j] = (numpydata[:,:,j] - meann[j] * 255) / (stdd[j] * 255)
    nordata = nordata.astype('float32')
    nordata = np.transpose(nordata, (2,0,1))
    nordata = nordata.reshape((1,3,32,32))
    nordata = torch.from_numpy(nordata)
    output = model(nordata)
    noo, preds = torch.max(output, 1)
    preds = preds.item()
    #preds = preds.numpy()
    print(preds)
    print(i)
    print(gt[i])
    result.append(preds==gt[i])
    #result.append(preds)
#print(result)
print(np.mean(result))
