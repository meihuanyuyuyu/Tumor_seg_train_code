import torch
from torch.utils.data import DataLoader,Dataset
import os
import cv2


class seg_dataset(Dataset):
    def __init__(self,transform=None) -> None:
        super().__init__()
        self.transf = transform
        self.imgs = os.listdir('train_dataset/img')
        self.labels = os.listdir('train_dataset/labels')

    def __getitem__(self, index):
        img = os.path.join('train_dataset/img',self.imgs[index])
        label = os.path.join('train_dataset/labels',self.labels[index])
        img = cv2.imread(img)
        label = cv2.imread(label,0)
        img = cv2.resize(img,[512,512]).transpose(2,0,1)[::-1].copy()
        label = cv2.resize(label,[512,512]).copy()
        label = (label!=0)*1.0
        img = (torch.from_numpy(img)/255).to(dtype=torch.float32)
        label = (torch.from_numpy(label)).to(dtype=torch.long).unsqueeze(0)
        return self.transform(img,label)
    

    def __len__(self):
        return len(self.imgs)

    def transform(self,img,label):
        if self.transf is not None:
            for _ in self.transf:
                img,label = _(img,label)
            return img,label
        else:
            return img,label



