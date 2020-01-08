import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np 
import argparse

from main_process import main
from IOtools import get_config_str



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset_setting')
    parser.add_argument('--dataset', default='SHB', help='choose dataset, SHA,SHB,QNRF')
    args = parser.parse_args()
    dataset_idxlist = {'SHA':0,'SHB':1,'QNRF':2}
    dataset_list = ['SH_partA','SH_partB','UCF-QNRF_ECCV18']    
    dataset_max = [[22],
                    [7],
                    [8]]
    dataset_choose = [dataset_idxlist[args.dataset] ]
    for di in dataset_choose:
        opt = dict()
        opt['dataset'] = dataset_list[di]
        opt['max_list'] = dataset_max[di]
        # step1: Create root path for dataset
        opt['root_dir'] = os.path.join(r'data',opt['dataset'])
        opt['num_workers'] = 0
        opt['IF_savemem_train'] = False
        opt['IF_savemem_test'] = False
        # -- test setting
        opt['test_batch_size'] = 1
        # --Network settinng    
        opt['psize'],opt['pstride'] = 64,64
        opt['div_times'] = 2
        # -- parse class to count setting
        parse_method_dict = {0:'maxp'}
        opt['parse_method'] = parse_method_dict[0]
        #step2: set the max number and partition method
        opt['max_num'] = opt['max_list'][0]
        opt['partition'] = 'two_linear'
        opt['step'] = 0.5
        # here create model path
        opt['model_path'] = os.path.join('model',args.dataset)
        main(opt)