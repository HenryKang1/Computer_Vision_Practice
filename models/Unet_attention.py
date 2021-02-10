import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

#attention unet

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi
class UNet(nn.Module):
    #Attention applied unet
    def __init__(self,img_ch=3,output_ch=1):
        super(UNet,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x) #[4, 64, 256, 256]

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4) #[4, 256, 64, 64]

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)               #[B,512,32,32]
        x4 = self.Att5(g=d5,x=x4)       #[B,512,32,32]
        d5 = torch.cat((x4,d5),dim=1)   #[B,1024,32,32]     
        d5 = self.Up_conv5(d5)          #[B,512,32,32]
        
        d4 = self.Up4(d5)               #[B,256,64,64]
        x3 = self.Att4(g=d4,x=x3)       #[4, 256, 64, 64]=[B,256,64,64],[4, 256, 64, 64]
        d4 = torch.cat((x3,d4),dim=1)   #[4, 512, 64, 64]
        d4 = self.Up_conv4(d4)          # [4, 256, 64, 64]

        d3 = self.Up3(d4)               # [4, 128, 128, 128] 
        x2 = self.Att3(g=d3,x=x2)       # [4, 128, 128, 128]
        d3 = torch.cat((x2,d3),dim=1)   #[4, 128, 128, 128]
        d3 = self.Up_conv3(d3)          #[4, 128, 128, 128]
        
        d2 = self.Up2(d3)             # [4, 64, 256, 256]
        x1 = self.Att2(g=d2,x=x1)     # [4, 64, 256, 256]
        d2 = torch.cat((x1,d2),dim=1) # [4,128, 256, 256]
        d2 = self.Up_conv2(d2)        ## [4,128, 256, 256]


        d1 = self.Conv_1x1(d2)        #[4, 1, 256, 256]

        
        return self.sigmoid(d1)
    @classmethod
    def load(cls, weights_path):
        print(f"Loading UNet from path `{weights_path}`")
        model = cls()
        model.load_state_dict(torch.load(weights_path))

        return model

    def save(self, save_path):
        torch.save(self.state_dict(), save_path)
        print(f"Saved model on path: {save_path}")



def unet(pretrained=False, **kwargs):
    model = UNet(**kwargs)
    #if pretrained:
    #    state_dict = torch.load('mobilenetv3_small_67.4.pth.tar') #model dict
    #    model.load_state_dict(state_dict, strict=True)

        # raise NotImplementedError
    return model

if __name__ == "__main__":
    #from etc.flops_counter import add_flops_counting_methods, flops_to_string, get_model_parameters_number

    model = UNet()

    # batch = torch.FloatTensor(1, 3, 480, 320)
    batch = torch.FloatTensor(1, 3, 512,512)

#    model_eval = add_flops_counting_methods(model)
#    model_eval.eval().start_flops_count()
#    out = model_eval(batch)  # ,only_encode=True)

#    print('Flops:  {}'.format(flops_to_string(model.compute_average_flops_cost())))
#    print('Params: ' + get_model_parameters_number(model))
#    print('Output shape: {}'.format(list(out.shape)))
#    total_paramters = sum(p.numel() for p in model.parameters())
#    print(total_paramters)

    import time

    use_gpu = True

    if use_gpu:
        model = model.cuda()  # .half()	#HALF seems to be doing slower for some reason
        batch = batch.cuda()  # .half()

    output = model(batch)
    print(output.shape)