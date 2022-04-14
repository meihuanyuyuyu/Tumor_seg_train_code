import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.resnet import Bottleneck


class conv_bn_act(nn.Module):
    def __init__(self,in_c,out_c) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_c,out_c,3,1,1)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.LeakyReLU(inplace=True)
    
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)


class bottleneck(nn.Module):
    def __init__(self,in_c,expansion=4) -> None:
        super().__init__()
        width = int(in_c/expansion)
        self.conv1 = nn.Conv2d(in_c,width,1,1)
        self.bn1 = nn.BatchNorm2d(in_c)
        self.conv2 = nn.Conv2d(width,width,3,1,1)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width,in_c,1,1)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += identity


        return out


class bottleneck_t(bottleneck):
    def __init__(self, in_c, expansion=4) -> None:
        super().__init__(in_c, expansion)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x),self.pool(x)




class Unet_pp(nn.Module):
    def __init__(self,num_classes) -> None:
        super().__init__()
        self.up30 = nn.ConvTranspose2d(512,256,2,2)
        self.up20 = nn.ConvTranspose2d(256,128,2,2)
        self.up10 = nn.ConvTranspose2d(128,64,2,2)

        self.up11 = nn.ConvTranspose2d(128,64,2,2)
        self.up21 = nn.ConvTranspose2d(256,128,2,2)

        self.up12 = nn.ConvTranspose2d(128,64,2,2)


        self.conv1 =nn.Sequential(
            conv_bn_act(3,64),
            bottleneck_t(64)
        )
        self.conv2 = nn.Sequential(
            conv_bn_act(64,128),
            bottleneck_t(128)
        )
        self.conv3 = nn.Sequential(
            conv_bn_act(128,256),
            bottleneck_t(256)
        )
        self.mid = nn.Sequential(
            conv_bn_act(256,512),
            bottleneck(512)
        )
        self.d01 = nn.Sequential(
            conv_bn_act(128,64),
            conv_bn_act(64,64),

        )
        self.d02 = nn.Sequential(
            conv_bn_act(192,64),
            conv_bn_act(64,64)            
        )
        self.d03=nn.Sequential(
            conv_bn_act(256,64),
            bottleneck(64)              
        )
        self.d11 = nn.Sequential(
            conv_bn_act(2*128,128),
            conv_bn_act(128,128)
        )
        self.d12 = nn.Sequential(
            conv_bn_act(3*128,128),
            conv_bn_act(128,128)           
        )
        self.d21 =nn.Sequential(
            conv_bn_act(2*256,256),
            conv_bn_act(256,256)           
        )
        for _ in range(3):
            final = nn.Conv2d(64,num_classes,1,1)
            setattr(self,f'final{_}',final)


    def forward(self,x):
        x0,x1 = self.conv1(x)
        x1,x2 = self.conv2(x1)
        x2,x3 = self.conv3(x2)
        x3 = self.mid(x3)

        x01 = self.d01(torch.cat([x0,self.up10(x1)],dim=1))
        x11 = self.d11(torch.cat([x1,self.up20(x2)],dim=1))
        x21 = self.d21(torch.cat([x2,self.up30(x3)],dim=1))
        
        x02 = self.d02(torch.cat([x0,self.up11(x11),x01],dim=1))
        x12 = self.d12(torch.cat([x1,self.up21(x21),x11],dim=1))

        x03 =  self.d03(torch.cat([x0,self.up12(x12),x01,x02],dim=1))
        return self.final0(x01),self.final1(x02),self.final2(x03)






class Unet_pp_width(nn.Module):
    r'nested unet model'
    def __init__(self,num_classes) -> None:
        super().__init__()
        self.up40 = nn.ConvTranspose2d(1024,512,2,2)
        self.up30 = nn.ConvTranspose2d(512,256,2,2)
        self.up20 = nn.ConvTranspose2d(256,128,2,2)
        self.up10 = nn.ConvTranspose2d(128,64,2,2)


        self.up11 = nn.ConvTranspose2d(128,64,2,2)
        self.up21 = nn.ConvTranspose2d(256,128,2,2)
        self.up31 = nn.ConvTranspose2d(512,256,2,2)

        self.up12 = nn.ConvTranspose2d(128,64,2,2)
        self.up22 = nn.ConvTranspose2d(256,128,2,2)

        self.up13 = nn.ConvTranspose2d(128,64,2,2)


        self.conv1 =nn.Sequential(
            conv_bn_act(3,64),
            bottleneck_t(64)
        )
        self.conv2 = nn.Sequential(
            conv_bn_act(64,128),
            bottleneck_t(128)
        )
        self.conv3 = nn.Sequential(
            conv_bn_act(128,256),
            bottleneck_t(256)
        )
        self.conv4 = nn.Sequential(
            conv_bn_act(256,512),
            bottleneck_t(512)
        )
        self.mid = nn.Sequential(
            conv_bn_act(512,1024),
            bottleneck(1024)
        )
        self.d01 = nn.Sequential(
            conv_bn_act(128,64),
            conv_bn_act(64,64),
            conv_bn_act(64,64)
        )
        self.d02 = nn.Sequential(
            conv_bn_act(192,64),
            conv_bn_act(64,64),
            conv_bn_act(64,64),             
        )
        self.d03=nn.Sequential(
            conv_bn_act(256,64),
            conv_bn_act(64,64), 
            conv_bn_act(64,64),              
        )
        self.d04 = nn.Sequential(
            conv_bn_act(5*64,64),
            conv_bn_act(64,64),
            conv_bn_act(64,64)               
        )
        self.d11 = nn.Sequential(
            conv_bn_act(2*128,128),
            conv_bn_act(128,128),
            conv_bn_act(128,128)
        )
        self.d12 = nn.Sequential(
            conv_bn_act(3*128,128),
            conv_bn_act(128,128),
            conv_bn_act(128,128)           
        )
        self.d13 = nn.Sequential(
            conv_bn_act(4*128,128),
            conv_bn_act(128,128),
            conv_bn_act(128,128)           
        )



        self.d21 =nn.Sequential(
            conv_bn_act(2*256,256),
            conv_bn_act(256,256),
            conv_bn_act(256,256)                       
        )
        self.d22 =nn.Sequential(
            conv_bn_act(3*256,256),
            conv_bn_act(256,256),
            conv_bn_act(256,256)                       
        )

        self.d31 = nn.Sequential(
            conv_bn_act(1024,512),
            conv_bn_act(512,512),
            conv_bn_act(512,512)                       
        )



        for _ in range(4):
            final = nn.Conv2d(64,num_classes,1,1)
            setattr(self,f'final{_}',final)


    def forward(self,x):
        x0,x1 = self.conv1(x)
        x1,x2 = self.conv2(x1)
        x2,x3 = self.conv3(x2)
        x3,x4 = self.conv4(x3)

        x4 = self.mid(x4)

        x01 = self.d01(torch.cat([x0,self.up10(x1)],dim=1))
        x11 = self.d11(torch.cat([x1,self.up20(x2)],dim=1))
        x21 = self.d21(torch.cat([x2,self.up30(x3)],dim=1))
        x31 = self.d31(torch.cat([x3,self.up40(x4)],dim=1))
        
        x02 = self.d02(torch.cat([x0,self.up11(x11),x01],dim=1))
        x12 = self.d12(torch.cat([x1,self.up21(x21),x11],dim=1))
        x22 = self.d22(torch.cat([x2,self.up31(x31),x21],dim=1))

        x03 =  self.d03(torch.cat([x0,self.up12(x12),x01,x02],dim=1))
        x13 = self.d13(torch.cat([x1,self.up22(x22),x11,x12],dim=1))

        x04 = self.d04(torch.cat([x0,self.up13(x13),x01,x02,x03],dim=1))
        
        final = (self.final0(x01)+ self.final1(x02)+self.final2(x03)+self.final3(x04))/4
        return final


