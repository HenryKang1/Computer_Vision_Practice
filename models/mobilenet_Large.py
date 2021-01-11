import time
import json
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.parallel 
import torch.backends.cudnn as cudnn
import torch.optim as optim
import os
import torch._utils
import torch.nn.functional as F
from torchvision.transforms import Normalize

def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 3, stride, 1, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )




class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        if nl == 'RE':
            nlin_layer = nn.ReLU # or ReLU6
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw
            conv_layer(inp, exp, 1, 1, 0, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),
            # dw
            conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False),
            norm_layer(exp),
            SELayer(exp),
            nlin_layer(inplace=True),
            # pw-linear
            conv_layer(exp, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        )


    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)



import torch.nn.functional as F

class decoder(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            # nn.Dropout2d(p=0.1, inplace=True),
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            # DANetHead(middle_channels, middle_channels), #144,144,48
            BaseOC(in_channels=middle_channels, out_channels=middle_channels,
                   key_channels=middle_channels // 2,
                   value_channels=middle_channels // 2,
                   dropout=0.2),
            # Parameters were chosen to avoid artifacts, suggested by https://distill.pub/2016/deconv-checkerboard/
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1),
            # upsample(scale_factor=2)
        )

    def forward(self, *args):
        x = torch.cat(args, 1)
        return self.block(x)
def upsample(size=None, scale_factor=None):
    return nn.Upsample(size=size, scale_factor=scale_factor, mode='bilinear', align_corners=False)


class ConvBnRelu(nn.Module):
    """Convenience layer combining a Conv2d, BatchNorm2d, and a ReLU activation.

    Original source of this code comes from
    https://github.com/lingtengqiu/Deeperlab-pytorch/blob/master/seg_opr/seg_oprs.py
    """
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 norm_layer=nn.BatchNorm2d):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = norm_layer(out_planes, eps=1e-5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class _LRASPP(nn.Module):
    """Lite R-ASPP"""

    def __init__(self, in_channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_LRASPP, self).__init__()
        out_channels = 128
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        )
        self.b1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(49, 49), stride=(16, 20),padding=(16, 20)),  # check it
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat2 = F.interpolate(feat2, size, mode='bilinear', align_corners=True)
        x = feat1 * feat2  # check it
        return x

class MobileNetV3(nn.Module):
    def __init__(self, n_class=1, input_size=512, dropout=0.8, width_mult=1.0):
        super().__init__()
        input_channel = 16

        # refer to Table 2 in paper

        mobile_setting = [
            [3, 16,  16,  False, 'RE', 1],
            [3, 64,  24,  False, 'RE', 2],
            [3, 72,  24,  False, 'RE', 1],
            [5, 72,  40,  True,  'RE', 2],
            [5, 120, 40,  True,  'RE', 1],
            [5, 120, 40,  True,  'RE', 1],

        ]
        mobile_setting1 = [
            # k, exp, c,  se,     nl,  s,
            [3, 240, 80,  False, 'HS', 2],
            [3, 200, 80,  False, 'HS', 1],
            [3, 184, 80,  False, 'HS', 1],
            [3, 184, 80,  False, 'HS', 1],
            [3, 480, 112, True,  'HS', 1],
            [3, 672, 112, True,  'HS', 1],
            ]

        mobile_setting3 = [
            # k, exp, c,  se,     nl,  s,
            [5, 672, 160, True,  'HS', 1],
            [5, 960, 160, True,  'HS', 1],
            [5, 960, 160, True,  'HS', 1],
        ]


        # building first layer
        assert input_size % 32 == 0
        self.features = conv_bn(3, input_channel, 2, nlin_layer=Hswish)
        self.features0 = nn.ModuleList()
        for k, exp, c, se, nl, s in mobile_setting:
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            self.features0.append(MobileBottleneck(input_channel, c, k, s, exp, se, nl))
            input_channel = output_channel

        # building mobile blocks
        self.features1 = nn.ModuleList()
        for k, exp, c, se, nl, s in mobile_setting1:
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            self.features1.append(MobileBottleneck(input_channel, c, k, s, exp, se, nl))
            input_channel = output_channel


        self.features3 = nn.ModuleList()
        for k, exp, c, se, nl, s in mobile_setting3:
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            self.features3.append(MobileBottleneck(input_channel, c, k, s, exp, se, nl))
            input_channel = output_channel
        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = Hswish()

        self.aspp=_LRASPP(960)
        self.cs4 = nn.Conv2d(40, 19, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(128, 19, kernel_size=1, stride=1, padding=0, bias=False)
        self.fconv= nn.Conv2d(16, n_class, kernel_size=1, stride=1, padding=0, bias=False)
        #self.sigmoid = nn.Sigmoid()
        self.conv4= ConvBnRelu(19+19, 19, kernel_size=1, stride=1, padding=0)  
    
        self.conv5 = ConvBnRelu(19+16, 16, kernel_size=1, stride=1, padding=0)  

    def forward(self, input):
        x = self.features(input)
        s2=x
        for i, layer in enumerate(self.features0):
            if i == 0:
                output2 = layer(x)
            else:
                output2 = layer(output2) # 1,24,256,256
        s4=output2
        for i, layer in enumerate(self.features1):
            if i == 0:
                output3 = layer(output2)
            else:
                output3 = layer(output3) # 1,24,256,256

        for i, layer in enumerate(self.features3):
            if i == 0:
                output4 = layer(output3)
            else:
                output4= layer(output4) # 1,96,64,64
        final = self.hs2(self.bn2(self.conv2(output4)))
        size = s4.size()[2:]

        aspp=self.aspp(final)
        aspp1 = F.interpolate(aspp, size, mode='bilinear', align_corners=True)
        coc=self.conv3(aspp1)
        s4=self.cs4(s4)
        y = torch.cat([coc, s4], 1)
        conc=self.conv4(y)
        #conc=coc+s4        
        size2 = s2.size()[2:]      
        final = F.interpolate(conc,size2, mode='bilinear', align_corners=True)
        y2 = torch.cat([final, s2], 1)
        final=self.conv5(y2)
        size3 = input.size()[2:]          
        final = F.interpolate(final,size3, mode='bilinear', align_corners=True)
        final=self.fconv(final)
        return final#self.sigmoid( final)  

    @classmethod
    def load(cls,weights_path):
        #print(f"Loading UNet from path `{weights_path}`")
        model = cls()
        model.load_state_dict(torch.load(weights_path))

        return model

    def save(self, save_path):
        torch.save(self.state_dict(), save_path)
        #print(f"Saved model on path: {save_path}")


def mobilenetv3(pretrained=False, **kwargs):
    model = MobileNetV3(**kwargs)
    #if pretrained:
    #    state_dict = torch.load('mobilenetv3_small_67.4.pth.tar') #model dict
    #    model.load_state_dict(state_dict, strict=True)

        # raise NotImplementedError
    return model
if __name__ == '__main__':
    net = MobileNetV3()
    #print('mobilenetv3:\n', net)
    print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))
    input_size=(1, 3,1024, 512)
    # pip install --upgrade git+https://github.com/kuan-wang/pytorch-OpCounter.git
    import datetime
    net=net.cuda()
    
    x = torch.randn(input_size)
    x=x.cuda()
    a = datetime.datetime.now()
    
    #net.eval()
    for i in range(100):
        out = net(x)
#        print(out.shape)

    b = datetime.datetime.now()
    c = b - a
    
    print(c)