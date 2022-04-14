from cProfile import label
import torch
from tqdm import tqdm
from model import Unet_pp, Unet_pp_width
import utils
import os
from torch import optim
from augmentation import Random_flip,Randomrotation_vh
from torchvision.utils import save_image
import dataset
from torch.utils.data import DataLoader,Subset
import numpy as np


train_indexes,test_indexes = utils.split_indexes('train_dataset/img')
data = dataset.seg_dataset([Random_flip(),Randomrotation_vh()])
train_data =data
test_data = Subset(data,test_indexes)
train_load = DataLoader(train_data,2,True,num_workers=4)
test_load = DataLoader(test_data,2,num_workers=4)

net = Unet_pp_width(2).to('cuda')
net.load_state_dict(torch.load('parameters/unet_pp_width_ds.pt'))
optimizer = optim.Adam(net.parameters(),lr=5e-6,weight_decay=1e-4)
dice_ce = utils.Dice_ce_loss(lamd=0.5)



for _ in range(30):
    dices1 = []
    #dices2 = []
    #dices3 = []
    bar = tqdm(train_load)
    net.train()
    for data_batch in bar:
        optimizer.zero_grad()
        imgs,labels = data_batch
        imgs = imgs.to('cuda')
        labels = labels.to('cuda')
        pred = net(imgs)
        #loss1 = dice_ce(pred_1,labels)
        #loss2 = dice_ce(pred_2,labels)
        #loss3 = dice_ce(pred_3,labels)
        loss = dice_ce(pred,labels)
        loss.backward()
        optimizer.step()
        bar.set_description(f'loss:{loss.item()}')
    
    with torch.no_grad():
        bar = tqdm(test_load)
        net.eval()
        for data_batch in bar:
            imgs,labels = data_batch
            imgs = imgs.to('cuda')
            labels = labels.to('cuda')
            pred_1 = net(imgs)
            pred1_bin = torch.argmax(pred_1,dim=1)
            #pred2_bin = torch.argmax(pred_2,dim=1)
            #pred3_bin = torch.argmax(pred_3,dim=1)
            dice_1 = utils.dice(pred1_bin,labels).item()
            #dice_2 = utils.dice(pred2_bin,labels).item()
            #dice_3 = utils.dice(pred3_bin,labels).item()
            dices1.append(dice_1)
            #dices2.append(dice_2)
            #dices3.append(dice_3)
            bar.set_description(f'dice1:{dice_1}')
        dices1 = np.array(dice_1).mean()
        #dices2 = np.array(dice_2).mean()
        #dices3 = np.array(dice_3).mean()
        result = torch.cat([labels,pred1_bin.unsqueeze(1)],dim=0).float()
        save_image(result,'visual_result.png')
        with open('exp_log_2.txt','a+') as f:
            f.write(f'dice1:{dices1};')
        torch.save(net.state_dict(),'parameters/unet_pp_width_ds.pt')


    



