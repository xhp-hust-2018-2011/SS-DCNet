#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: xionghaipeng
"""

__author__='xhp'

'''load the dataset'''
#from __future__ import print_function, division
import os
import torch
import numpy as np
import glob#use glob.glob to get special flielist
import scipy.io as sio#use to import mat as dic,data is ndarray
# load image
from PIL import Image
# torch related
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from torchvision import transforms
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")



class Countmap_Dataset(Dataset):
    """Wheat dataset. also can be used for annotation like density map"""

    def __init__(self, img_dir,tar_dir, rgb_dir,transform=None,if_test = False,\
        IF_loadmem=False):
        """
        Args:
            img_dir (string ): Directory with all the images.
            tar_dir (string ): Path to the annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.IF_loadmem = IF_loadmem #whether to load data in memory
        self.IF_loadFinished = False
        self.image_mem = []
        self.target_mem = []
        self.img_dir = img_dir
        self.tar_dir = tar_dir
        self.transform = transform
        mat = sio.loadmat(rgb_dir)
        self.rgb = mat['rgbMean'].reshape(1,1,3) 
        img_name = os.path.join(self.img_dir,'*.jpg')
        self.filelist =  glob.glob(img_name)
        self.dataset_len = len(self.filelist)
        # for test process, load data is different
        self.if_test = if_test
        self.DIV = 64 # pad the orignial image to be divisible by 64

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        # ------------------------------------
        # 1. see if load from disk or memory
        # ------------------------------------
        if (not self.IF_loadmem) or (not self.IF_loadFinished): 
            # load image
            img_name = self.filelist[idx]
            image = Image.open(img_name).convert('RGB')
            image = transforms.ToTensor()(image)
            image = get_pad(image,DIV=64)
            image = image - torch.Tensor(self.rgb).view(3,1,1)           
            (filepath,tempfilename) = os.path.split(img_name)
            (name,extension) = os.path.splitext(tempfilename)
            # load gt
            mat_dir = os.path.join( self.tar_dir, '%s.mat' % (name) )
            mat = sio.loadmat(mat_dir)
            # if need to save in memory
            if self.IF_loadmem:
                self.image_mem.append(image)
                self.target_mem.append(mat)
                # updata if load finished
                if len(self.image_mem) == self.dataset_len:
                    self.IF_loadFinished = True
        else:
            image = self.image_mem[idx]
            mat = self.target_mem[idx]
        # collect to sample
        all_num = mat['dot_num'].reshape((1,1)) # use dot number as counts for testing !!!
        sample = {'image': image,'all_num':all_num}
        # GT To Tensor
        sample['all_num'] = torch.from_numpy(sample['all_num'])
        # tranformation to tensor
        if self.transform:
            for t in self.transform:
                sample = t(sample)
        sample['name'] = name
        return sample
    

def get_pad(inputs,DIV=64):
    h,w = inputs.size()[-2:]
    ph,pw = (DIV-h%DIV),(DIV-w%DIV)
    # print(ph,pw)

    tmp_pad = [0,0,0,0]
    if (ph!=DIV): 
        tmp_pad[2],tmp_pad[3] = 0,ph
    if (pw!=DIV):
        tmp_pad[0],tmp_pad[1] = 0,pw
        
    # print(tmp_pad)
    inputs = F.pad(inputs,tmp_pad)

    return inputs

if __name__ =='__main__':
    inputs = torch.ones(6,60,730,970);print('ori_input_size:',str(inputs.size()) )
    inputs = get_pad(inputs);print('pad_input_size:',str(inputs.size()) )
