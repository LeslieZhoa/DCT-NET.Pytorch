#! /usr/bin/python 
# -*- encoding: utf-8 -*-
'''
@author LeslieZhao
@date 20220721
'''


from torch.utils.data import Dataset
import torch.distributed as dist


class DatasetBase(Dataset):
    def __init__(self,slice_id=0,slice_count=1,use_dist=False,**kwargs):

        if use_dist:
            slice_id = dist.get_rank()
            slice_count = dist.get_world_size()
        self.id = slice_id 
        self.count = slice_count


    def __getitem__(self,i):
        pass

        


    def __len__(self):
        return 1000

