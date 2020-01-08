import torch.nn as nn
import torch
from torchvision import models

import torch.nn.functional as F
import math

try:
    from class_func import Class2Count
    from merge_func import count_merge_low2high_batch
    from base_Network_module import up,up_res
except:
    from Network.class_func import Class2Count
    from Network.merge_func import count_merge_low2high_batch
    from Network.base_Network_module import up,up_res
# ============================================================================
# 1.base module
# ============================================================================ 

def Gauss_initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


# --1.1 
def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)   

# --define base Netweork module
class VGG16_frontend(nn.Module):
    def __init__(self,block_num=5,decode_num=0,load_weights=True,bn=False,IF_freeze_bn=False):
        super(VGG16_frontend,self).__init__()
        self.block_num = block_num
        self.load_weights = load_weights
        self.bn = bn
        self.IF_freeze_bn = IF_freeze_bn
        self.decode_num = decode_num

        block_dict = [[64, 64, 'M'], [128, 128, 'M'], [256, 256, 256, 'M'],\
             [512, 512, 512,'M'], [512, 512, 512,'M']]

        self.frontend_feat = []
        for i in range(block_num):
            self.frontend_feat += block_dict[i]

        if self.bn:
            self.features = make_layers(self.frontend_feat, batch_norm=True)
        else:
            self.features = make_layers(self.frontend_feat, batch_norm=False)


        if self.load_weights:
            if self.bn:
                pretrained_model = models.vgg16_bn(pretrained = True)
            else:
                pretrained_model = models.vgg16(pretrained = True)
            pretrained_dict = pretrained_model.state_dict()
            model_dict = self.state_dict()
            # filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # overwrite entries in the existing state dict
            model_dict.update(pretrained_dict) 
            # load the new state dict
            self.load_state_dict(model_dict)

        if IF_freeze_bn:
            self.freeze_bn()

    def forward(self,x):
        if self.bn: 
            x = self.features[ 0: 7](x)
            conv1_feat =x if self.decode_num>=4 else []
            x = self.features[ 7:14](x)
            conv2_feat =x if self.decode_num>=3 else []
            x = self.features[ 14:24](x)
            conv3_feat =x if self.decode_num>=2 else []
            x = self.features[ 24:34](x)
            conv4_feat =x if self.decode_num>=1 else []
            x = self.features[ 34:44](x)
            conv5_feat =x 
        else:
            x = self.features[ 0: 5](x)
            conv1_feat =x if self.decode_num>=4 else []
            x = self.features[ 5:10](x)
            conv2_feat =x if self.decode_num>=3 else []
            x = self.features[ 10:17](x)
            conv3_feat =x if self.decode_num>=2 else []
            x = self.features[ 17:24](x)
            conv4_feat =x if self.decode_num>=1 else []
            x = self.features[ 24:31](x)
            conv5_feat =x 
               
        feature_map = {'conv1':conv1_feat,'conv2': conv2_feat,\
            'conv3':conv3_feat,'conv4': conv4_feat, 'conv5': conv5_feat}   
        
        return feature_map


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()



class SSDCNet_classify(nn.Module):
    def __init__(self, class_num,label_indice,div_times=2,\
        frontend_name='VGG16',block_num=5,\
        IF_pre_bn=True,IF_freeze_bn=False,load_weights=False,\
        psize=64,pstride=64,parse_method ='maxp',den_for='p'):
        super(SSDCNet_classify, self).__init__()

        # init parameters
        self.label_indice = label_indice # this should be tensor
        self.class_num = len(self.label_indice)+1
        self.div_times = div_times
        self.frontend_name = frontend_name
        self.block_num = block_num

        self.IF_pre_bn = IF_pre_bn
        self.IF_freeze_bn = IF_freeze_bn
        self.load_weights = load_weights
        self.psize,self.pstride = psize,pstride
        self.parse_method = parse_method
        self.den_for = den_for

        # first, make frontend
        if self.frontend_name == 'VGG16':
            self.front_end = VGG16_frontend(block_num=self.block_num,decode_num=self.div_times,
            load_weights=self.load_weights,bn=self.IF_pre_bn,IF_freeze_bn=self.IF_freeze_bn)
            self.back_end_up = dict()
            # use light wight Refinet upsample
            up_in_ch = [512,512,256,128,64]
            up_out_ch = [256,256,128,64]
            cat_in_ch = [(256+512),(256+256)]
            cat_out_ch = [512,512]
            
            self.back_end_up = dict()
            back_end_up = []
            for i in range(self.div_times):
                back_end_up.append( up(up_in_ch[i],up_out_ch[i],\
                    cat_in_ch[i],cat_out_ch[i]) )               

            if self.div_times>0:
                self.back_end_up = nn.Sequential(*back_end_up)
        
        # make backend pre
        self.back_end_cls = torch.nn.Sequential(
                torch.nn.AvgPool2d((2,2),stride=2),
                torch.nn.Conv2d(512, 512, (1, 1) ),
                torch.nn.ReLU(),
                torch.nn.Conv2d(512, class_num, (1, 1) ) ) 

        self.back_end_lw_fc = torch.nn.Sequential(
                torch.nn.AvgPool2d((2,2),stride=2),
                torch.nn.Conv2d(512, 512, (1, 1) ),
                torch.nn.ReLU(),
                torch.nn.Conv2d(512, 1, (1, 1) ) ) 

        # 2019/09/12 add density map predictor
        # 2x larger density map than class map
        
        self.back_end_denisty = torch.nn.Sequential(
                torch.nn.Conv2d(512, 512, (1, 1) ),
                torch.nn.ReLU(),
                torch.nn.Conv2d(512, 1, (1, 1) ) ) 
        

        Gauss_initialize_weights(self.back_end_up) 
        Gauss_initialize_weights(self.back_end_cls)
        Gauss_initialize_weights(self.back_end_lw_fc)
        Gauss_initialize_weights(self.back_end_denisty)
        
   

    def forward(self,x):
        x = self.front_end(x)
        return x

    def resample(self,feature_map):

        low_feat  = feature_map['conv5']

        div_res = dict()
        div_cls_name = 'cls' + str(0)
        new_conv_reg = self.back_end_cls(low_feat) 
        div_res[div_cls_name] = new_conv_reg


        for i in range(self.div_times):
            # low feat to create density maps
            feat_h,feat_w =  low_feat.size()[-2:]
            tmp_density = self.back_end_denisty(low_feat)
            tmp_density = F.unfold(tmp_density,kernel_size = 2,stride=2)
            # tmp_density = F.unfold(low_feat.sum(dim=1,keepdim=True),kernel_size = 2,stride=2)
            tmp_density = F.softmax(tmp_density,dim=1)
            tmp_density = F.fold(tmp_density,(feat_h,feat_w),kernel_size=2,stride=2)
            tmp_density_name = 'den' + str(i)
            div_res[tmp_density_name] = tmp_density
            
            high_feat_name = 'conv'+str(4-i)
            high_feat = feature_map[high_feat_name]
            low_feat = self.back_end_up[i](low_feat,high_feat)


            # div45: Upsample and get weight  
            new_conv_w = self.back_end_lw_fc(low_feat)
            # new_conv4_w = F.sigmoid(new_conv4_w)
            new_conv_w = torch.sigmoid(new_conv_w)
            new_conv_reg = self.back_end_cls(low_feat) 

            del feature_map[high_feat_name]

            div_w_name = 'w' + str(i+1)
            div_cls_name = 'cls' + str(i+1)

            div_res[div_w_name] = new_conv_w
            div_res[div_cls_name] = new_conv_reg
            
        del feature_map
        return div_res

    def parse_merge(self,div_res):
        res = dict()
        # class2count
        for cidx in range(self.div_times+1):
            tname = 'c' + str(cidx)

            if self.parse_method == 'maxp':
                div_res['cls'+str(cidx)] = div_res['cls'+str(cidx)].max(dim=1,keepdim=True)[1]
                res[tname] = Class2Count(div_res['cls'+str(cidx)],self.label_indice)
            elif self.parse_method == 'mulp':
                div_res['cls'+str(cidx)] = F.softmax(div_res['cls'+str(cidx)],dim=1)
                res[tname] = Class2Count_mul(div_res['cls'+str(cidx)],self.label_indice)

        # merge_div_res
        # res['c0'] is the parse result
        res['div0'] = res['c0'] 
        for divt in range(1,self.div_times+1):
            den_name = 'den' + str(divt-1)
            tden = div_res[den_name]

            tname = 'div' + str(divt) 
            tchigh = res['c' + str(divt)] 
            tclow = res['div' + str(int(divt-1))]

            IF_p = True if self.den_for == 'p' else False
            tclow = count_merge_low2high_batch(tclow,tden,IF_avg=False,IF_p=IF_p)
            tw = div_res['w'+str(divt)]
            res[tname] = (1-tw)*tclow + tw*tchigh

            # save div map
            res['w'+str(divt)] = tw
        
        del div_res
        return res    
        

    


