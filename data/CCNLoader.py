#! /usr/bin/python 
# -*- encoding: utf-8 -*-
'''
@author LeslieZhao
@date 20220721
'''

import os 

from torchvision import transforms 
import PIL.Image as Image
from data.DataLoader import DatasetBase
import random


class CCNData(DatasetBase):
    def __init__(self, slice_id=0, slice_count=1,dist=False, **kwargs):
        super().__init__(slice_id, slice_count,dist, **kwargs)


        self.transform = transforms.Compose([
             transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
       
        root = kwargs['root']
        self.paths = [os.path.join(root,f) for f in os.listdir(root)]
        self.length = len(self.paths)
        random.shuffle(self.paths)

    def __getitem__(self,i):
        idx = i % self.length
        img_path = self.paths[idx]

        with Image.open(img_path) as img:
            Img = self.transform(img)

        return Img


    def __len__(self):
        return max(100000,self.length)
        # return 4

