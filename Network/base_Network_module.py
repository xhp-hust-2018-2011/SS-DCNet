import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --1.2.1
class one_conv(nn.Module):
    def __init__(self, in_ch, out_ch, normaliz=False):
        super(one_conv, self).__init__()

        ops = []
        ops += [nn.Conv2d(in_ch, out_ch, 3, padding=1)]
        # ops += [nn.Dropout(p=0.1)]
        if normaliz:
            ops += [nn.BatchNorm2d(out_ch)]
        ops += [nn.ReLU(inplace=True)]

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

# --1.2.2    
class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch, normaliz=False):
        super(double_conv, self).__init__()

        ops = []
        ops += [nn.Conv2d(in_ch, out_ch, 3, padding=1)]
        # ops += [nn.Dropout(p=0.1)]
        if normaliz:
            ops += [nn.BatchNorm2d(out_ch)]
        ops += [nn.ReLU(inplace=True)]
        ops += [nn.Conv2d(out_ch, out_ch, 3, padding=1)]
        # ops += [nn.Dropout(p=0.1)]
        if normaliz:
            ops += [nn.BatchNorm2d(out_ch)]
        ops += [nn.ReLU(inplace=True)]

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

# --1.2.3
class three_conv(nn.Module):
    def __init__(self, in_ch, out_ch, normaliz=False):
        super(three_conv, self).__init__()

        ops = []
        ops += [nn.Conv2d(in_ch, out_ch, 3, padding=1)]
        # ops += [nn.Dropout(p=0.1)]
        if normaliz:
            ops += [nn.BatchNorm2d(out_ch)]
        ops += [nn.ReLU(inplace=True)]
        
        ops += [nn.Conv2d(out_ch, out_ch, 3, padding=1)]
        # ops += [nn.Dropout(p=0.1)]
        if normaliz:
            ops += [nn.BatchNorm2d(out_ch)]
        ops += [nn.ReLU(inplace=True)]
        
        ops += [nn.Conv2d(out_ch, out_ch, 3, padding=1)]
        # ops += [nn.Dropout(p=0.1)]
        if normaliz:
            ops += [nn.BatchNorm2d(out_ch)]
        ops += [nn.ReLU(inplace=True)]

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class resconv2(nn.Module):
    def __init__(self,in_ch,out_ch,ksize=3,kstride=1,kpad=1):
        super(resconv2,self).__init__()
        self.conv1 = nn.Conv2d(in_ch,out_ch,ksize,stride=kstride,padding=kpad)
        self.conv2 = nn.Conv2d(out_ch,out_ch,ksize,stride=kstride,padding=kpad) 
        if in_ch != out_ch:
            self.red = nn.Conv2d(in_ch,out_ch,(1,1),stride=1,padding=0)
        else:
            self.red = None

    def forward(self,x):
        rx = self.conv1(x)
        rx = F.relu(rx)
        rx= self.conv2(rx)
        rx = F.relu(rx)

        if self.red!=None:
            x = self.red(x)+rx
        else:
            x = x + rx
        return rx

class up_res(nn.Module):
    def __init__(self, up_in_ch, up_out_ch,cat_in_ch, cat_out_ch,if_convt=False):
        super(up_res, self).__init__()
        self.if_convt = if_convt
        if self.if_convt:
            self.up = nn.ConvTranspose2d(up_in_ch,up_out_ch, 2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2,
                                mode='bilinear',
                                align_corners=False)

            self.conv1 = nn.Conv2d(up_in_ch,up_out_ch,(3,3))
     
        self.conv2 = resconv2(cat_in_ch,cat_out_ch)

    def forward(self, x1, x2):

        if self.if_convt:
            x1 = self.up(x1)
        else:
            x1 = self.up(x1)
            x1 = self.conv1(x1)
            
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        #pad to make up for the loss when downsampling
        x1 = F.pad(x1, (diffX // 2, int(math.ceil(diffX / 2.0)),
                        diffY // 2, int(math.ceil(diffY / 2.0))))#3//2=1,3/2=1.5
        x = torch.cat([x2, x1], dim=1)
        del x2,x1
        x = self.conv2(x)
        return x


# --1.3.1
class up(nn.Module):
    def __init__(self, up_in_ch, up_out_ch,cat_in_ch, cat_out_ch,if_convt=False):
        super(up, self).__init__()
        self.if_convt = if_convt
        if self.if_convt:
            self.up = nn.ConvTranspose2d(up_in_ch,up_out_ch, 2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2,
                                mode='bilinear',
                                align_corners=False)
            self.conv1 = one_conv(up_in_ch,up_out_ch)
     
        self.conv2 = double_conv(cat_in_ch, cat_out_ch)

    def forward(self, x1, x2):

        if self.if_convt:
            x1 = self.up(x1)
        else:
            x1 = self.up(x1)
            x1 = self.conv1(x1)
            
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        #pad to make up for the loss when downsampling
        x1 = F.pad(x1, (diffX // 2, int(math.ceil(diffX / 2.0)),
                        diffY // 2, int(math.ceil(diffY / 2.0))))#3//2=1,3/2=1.5
        x = torch.cat([x2, x1], dim=1)
        del x2,x1
        x = self.conv2(x)
        return x

# --1.3.2
class upcat(nn.Module):
    def __init__(self, up_in_ch, up_out_ch,if_convt=False):
        super(upcat, self).__init__()
        self.if_convt = if_convt
        if self.if_convt:
            self.up = nn.ConvTranspose2d(up_in_ch, up_out_ch, 2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2,
                                mode='bilinear',
                                align_corners=False)
            self.conv1 = one_conv(up_in_ch,up_out_ch)
     
    def forward(self, x1, x2):

        if self.if_convt:
            x1 = self.up(x1)
        else:
            x1 = self.up(x1)
            x1 = self.conv1(x1)
            
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        #pad to make up for the loss when downsampling
        x1 = F.pad(x1, (diffX // 2, int(math.ceil(diffX / 2.0)),
                        diffY // 2, int(math.ceil(diffY / 2.0))))#3//2=1,3/2=1.5
        x = torch.cat([x2, x1], dim=1)
        del x2,x1

        return x

# --1.4
def change_padding(net,del_or_add='del',pad_size=(1,1)):
    for m in net.modules():
        if isinstance(m,nn.Conv2d):
            m.padding = (0,0) if del_or_add =='del' else pad_size

    return net

# --1.5 can only compute linear
def compute_rf(net):
    rf_size,rf_pad,rf_stride = 1,0,1
    for m in net.modules():
        if isinstance(m,(nn.Conv2d,nn.MaxPool2d)):
            tmp_kernel_size = m.kernel_size[0] if isinstance(m.kernel_size,(tuple,list)) else m.kernel_size
            tmp_padding = m.padding[0] if isinstance(m.padding,(tuple,list)) else m.padding
            tmp_stride = m.stride[0] if isinstance(m.stride,(tuple,list)) else m.stride
            
            # rf_pad relates with the last layer's rf_stride
            rf_pad += tmp_padding*rf_stride
            # rf_size relates with the last layers's rf_stride
            rf_size += (tmp_kernel_size-1)*rf_stride
            rf_stride *= tmp_stride

    return {'rf_size':rf_size,'rf_pad':rf_pad,'rf_stride':rf_stride}