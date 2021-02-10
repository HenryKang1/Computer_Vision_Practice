#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn

import torch.nn.functional as F
import torchvision


class Bottleneck(nn.Module):
    def __init__(self,
            in_chan,
            out_chan,
            stride = 1,
            stride_at_1x1 = False,
            dilation = 1,
            *args, **kwargs):
        super(Bottleneck, self).__init__(*args, **kwargs)

        stride1x1, stride3x3 = (stride, 1) if stride_at_1x1 else (1, stride)
        assert out_chan % 4 == 0
        mid_chan = int(out_chan / 4)

        self.conv1 = nn.Conv2d(in_chan,
                mid_chan,
                kernel_size = 1,
                stride = stride1x1,
                bias = False)
        self.bn1 = nn.BatchNorm2d(mid_chan)
        self.conv2 = nn.Conv2d(mid_chan,
                mid_chan,
                kernel_size = 3,
                stride = stride3x3,
                padding = dilation,
                dilation = dilation,
                bias = False)
        self.bn2 = nn.BatchNorm2d(mid_chan)
        self.conv3 = nn.Conv2d(mid_chan,
                out_chan,
                kernel_size=1,
                bias=False)
        self.bn3 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chan))
        self.init_weight()

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        residual = self.conv3(residual)
        residual = self.bn3(residual)

        if self.downsample == None:
            inten = x
        else:
            inten = self.downsample(x)
        out = residual + inten
        out = self.relu(out)

        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


def create_stage(in_chan, out_chan, b_num, stride=1, dilation=1):
    assert out_chan % 4 == 0
    mid_chan = out_chan / 4
    blocks = [Bottleneck(in_chan, out_chan, stride=stride, dilation=dilation),]
    for i in range(1, b_num):
        blocks.append(Bottleneck(out_chan, out_chan, stride=1, dilation=dilation))
    return nn.Sequential(*blocks)


class Resnet101(nn.Module):
    def __init__(self, stride=32, *args, **kwargs):
        super(Resnet101, self).__init__()
        assert stride in (8, 16, 32)
        dils = [1, 1] if stride==32 else [el*(16//stride) for el in (1, 2)]
        strds = [2 if el==1 else 1 for el in dils]

        self.conv1 = nn.Conv2d(
                3,
                64,
                kernel_size = 7,
                stride = 2,
                padding = 3,
                bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(
                kernel_size = 3,
                stride = 2,
                padding = 1,
                dilation = 1,
                ceil_mode = False)
        self.layer1 = create_stage(64, 256, 3, stride=1, dilation=1)
        self.layer2 = create_stage(256, 512, 4, stride=2, dilation=1)
        self.layer3 = create_stage(512, 1024, 23, stride=strds[0], dilation=dils[0])
        self.layer4 = create_stage(1024, 2048, 3, stride=strds[1], dilation=dils[1])


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        feat4 = self.layer1(x)
        feat8 = self.layer2(feat4)
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)
        return feat4, feat8, feat16, feat32

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, dilation=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                dilation = dilation,
                bias = True)
        self.bn = nn.BatchNorm2d(out_chan)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class ASPP(nn.Module):
    def __init__(self, in_chan=2048, out_chan=256, with_gp=True):
        super(ASPP, self).__init__()
        self.with_gp = with_gp
        self.conv1 = ConvBNReLU(in_chan, out_chan, ks=1, dilation=1, padding=0)
        self.conv2 = ConvBNReLU(in_chan, out_chan, ks=3, dilation=2, padding=2)
        self.conv3 = ConvBNReLU(in_chan, out_chan, ks=3, dilation=4, padding=4)
        self.conv4 = ConvBNReLU(in_chan, out_chan, ks=3, dilation=8, padding=8)
        if self.with_gp:
            self.avg = nn.AdaptiveAvgPool2d((1, 1))
            self.conv1x1 = ConvBNReLU(in_chan, out_chan, ks=1)
            self.conv_out = ConvBNReLU(out_chan*5, out_chan, ks=1)
        else:
            self.conv_out = ConvBNReLU(out_chan*4, out_chan, ks=1)

        #self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        if self.with_gp:
            avg = self.avg(x)
            feat5 = self.conv1x1(avg)
            feat5 = F.interpolate(feat5, (H, W), mode='bilinear', align_corners=True)
            feat = torch.cat([feat1, feat2, feat3, feat4, feat5], 1)
        else:
            feat = torch.cat([feat1, feat2, feat3, feat4], 1)
        feat = self.conv_out(feat)
        return feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class Decoder(nn.Module):
    def __init__(self, classes=1, low_chan=256):
        super(Decoder, self).__init__()
        self.conv_low = ConvBNReLU(low_chan, 48, ks=1, padding=0)
        self.conv_cat = nn.Sequential(
                ConvBNReLU(304, 256, ks=3, padding=1),
                ConvBNReLU(256, 256, ks=3, padding=1),
                )
        self.conv_out = nn.Conv2d(256, classes, kernel_size=1, bias=False)

        #self.init_weight()

    def forward(self, feat_low, feat_aspp):
        H, W = feat_low.size()[2:]
        feat_low = self.conv_low(feat_low)
        feat_aspp_up = F.interpolate(feat_aspp, (H, W), mode='bilinear',align_corners=True)
        feat_cat = torch.cat([feat_low, feat_aspp_up], dim=1)
        feat_out = self.conv_cat(feat_cat)
        logits = self.conv_out(feat_out)
        return logits

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class Deeplab_v3plus(nn.Module):
    def __init__(self,classes=1):
        super().__init__()
        self.backbone = Resnet101(stride=16)
        self.aspp = ASPP(in_chan=2048, out_chan=256)
        self.decoder = Decoder(classes=1, low_chan=256)
        #  self.backbone = Darknet53(stride=16)
        #  self.aspp = ASPP(in_chan=1024, out_chan=256, with_gp=False)
        #  self.decoder = Decoder(cfg.n_classes, low_chan=128)

        #self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feat4, _, _, feat32 = self.backbone(x)
        feat_aspp = self.aspp(feat32)
        logits = self.decoder(feat4, feat_aspp)
        logits = F.interpolate(logits, (H, W), mode='bilinear', align_corners=True)

        return logits

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
    @classmethod
    def load(cls,weights_path):
        #print(f"Loading UNet from path `{weights_path}`")
        model = cls()
        model.load_state_dict(torch.load(weights_path))

        return model

    def save(self, save_path):
        torch.save(self.state_dict(), save_path)
        #print(f"Saved model on path: {save_path}")



if __name__ == "__main__":
    net = Deeplab_v3plus()
    net.cuda()
    net.train()
    net = nn.DataParallel(net)
    for i in range(100):
        #  with torch.no_grad():
        in_ten = torch.randn((1, 3, 768, 768)).cuda()
        logits = net(in_ten)
        print(i)
        print(logits.size())
