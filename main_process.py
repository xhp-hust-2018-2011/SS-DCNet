import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import  DataLoader
import torch.optim as optim

import os
import numpy as np
from time import time
import math
import pandas as pd
import csv

from IOtools import txt_write,get_config_str
from load_data_V2 import Countmap_Dataset
from Network.SSDCNet import SSDCNet_classify
from Val import test_phase

def main(opt):
    # save folder
    save_folder = opt['model_path'] 
    # =============================================================================
    # inital setting
    # =============================================================================
    # 1.Initial setting
    # --1.1 dataset setting
    dataset = opt['dataset']
    root_dir = opt['root_dir']
    num_workers = opt['num_workers']
    img_subsubdir = 'images'; tar_subsubdir = 'gtdens'
    transform_test = []
    # --1.3 use initial setting to generate
    # set label_indice
    label_indice = np.arange(opt['step'],opt['max_num']+opt['step'],opt['step'])
    add = np.array([1e-6,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45]) 
    label_indice = np.concatenate( (add,label_indice) )
    print(label_indice)
    opt['label_indice'] = label_indice
    opt['class_num'] = label_indice.size+1
    #test settings
    img_dir = os.path.join(root_dir,'test',img_subsubdir)
    tar_dir = os.path.join(root_dir,'test',tar_subsubdir)
    rgb_dir = os.path.join(root_dir,'rgbstate.mat')
    testset = Countmap_Dataset(img_dir,tar_dir,rgb_dir,transform=transform_test,\
        if_test=True, IF_loadmem=opt['IF_savemem_test'])
    testloader = DataLoader(testset, batch_size=opt['test_batch_size'],
                            shuffle=False, num_workers=num_workers)
    # init networks
    label_indice = torch.Tensor(label_indice)
    class_num = len(label_indice)+1
    div_times = opt['div_times']
    net = SSDCNet_classify(class_num,label_indice,div_times=div_times,\
            frontend_name='VGG16',block_num=5,\
            IF_pre_bn=False,IF_freeze_bn=False,load_weights=True,\
            psize=opt['psize'],pstride = opt['pstride'],parse_method ='maxp').cuda()
    # test the min epoch
    mod_path='best_epoch.pth' 
    mod_path=os.path.join(opt['model_path'] ,mod_path)
    if os.path.exists(mod_path):
        all_state_dict = torch.load(mod_path)
        net.load_state_dict(all_state_dict['net_state_dict'])
        tmp_epoch_num = all_state_dict['tmp_epoch_num']
        log_save_path = os.path.join(save_folder,'log-epoch-min[%d]-%s.txt' \
            %(tmp_epoch_num+1,opt['parse_method']) )
        # test
        test_log = test_phase(opt,net,testloader,log_save_path=log_save_path)