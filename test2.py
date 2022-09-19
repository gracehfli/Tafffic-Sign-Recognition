import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch                                                                     
import os                                                                        
import albumentations as A                                                       
from albumentations.pytorch import ToTensorV2                                    
from torch.nn import functional as F                                             
from model import Net


device = 'cpu'
model = Net().to(device)
model = model.eval()
model.load_state_dict(
            torch.load(
                        '../outputs/model.pth', map_location=device
                            )['model_state_dict']
            )

def pre_image(image_path,model):
    img = Image.open(image_path)
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225]
    transform_norm = transforms.Compose([transforms.ToTensor(), 
    transforms.Resize((32,32)),transforms.Normalize(mean,std)])
    # get normalized image
    img_normalized = transform_norm(img)
    #print(img_normalized)
    img_normalized = img_normalized.unsqueeze_(0)
    # input = Variable(image_tensor)
    img_normalized = img_normalized.to(device)
    print(img_normalized)
    print(img_normalized.shape)
    print(img_normalized.dtype)
    with torch.no_grad():
        model.eval()  
        output =model(img_normalized)
        noo, preds = torch.max(output.data, 1)
        return preds

predict =pre_image("/Users/lihongfei/Desktop/CS231n/input/GTSRB_Final_Test_Images/GTSRB/Final_Test/images/00000.ppm",model)
print(predict)
