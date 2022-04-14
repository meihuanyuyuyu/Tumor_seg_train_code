import cv2
import torch
from PIL import Image
import torchvision.transforms.functional as ttf
from torchvision.utils import save_image

#img = cv2.imread('train_dataset/img/6.png')
label = cv2.imread('train_dataset/labels/6.png',0)
#img = cv2.resize(img,[512,512]).transpose(2,0,1)[::-1].copy()
label = cv2.resize(label,[512,512]).copy()
print(label.shape)
#img = (torch.from_numpy(img)/255).to(dtype=torch.float32)
label = (torch.from_numpy(label)).to(dtype=torch.float32)
label = (label!=0)*1.0
save_image(label,'img.png')
print(label)
print(label.shape,label.max())