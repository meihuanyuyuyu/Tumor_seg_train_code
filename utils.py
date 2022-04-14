import torch 
import torch.nn as nn
import os
import torch.nn.functional as nnf
import torchvision.transforms as tf


class Dice_ce_loss(nn.Module):
    def __init__(self,smooth=0.001,lamd=0.5) -> None:
        super().__init__()
        self.smooth =smooth
        self.lamd = lamd

    def _dice(self,pred,target):
        pred = nnf.softmax(pred,dim=1)
        pred = torch.max(pred,dim=1).values
        dice = (2*(pred*target).sum()+self.smooth)/(pred.sum()+target.sum()+self.smooth)
        return 1-dice
    
    def forward(self,pred,target):
        target=target.squeeze(1)
        return self.lamd * self._dice(pred,target) + (1-self.lamd)* nnf.cross_entropy(pred,target,label_smoothing=0.001)


def split_indexes(fp:str):
    imgs = os.listdir(fp)
    length = len(imgs)
    return torch.randperm(length).split(int(0.8*length))


def iou(pred_bin,target):
    intersact = (pred_bin.bool() & target.bool()).sum()
    union = (pred_bin.bool() | target.bool()).sum()
    return intersact/union

def dice(pred_bin,target):
    tp = 2*(pred_bin*target).sum()
    tp_fn_fp =pred_bin.sum()+target.sum()
    return tp/tp_fn_fp
