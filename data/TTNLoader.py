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
import numpy as np
import torch


class TTNData(DatasetBase):
    def __init__(self, slice_id=0, slice_count=1,dist=False, **kwargs):
        super().__init__(slice_id, slice_count,dist, **kwargs)


        self.transform = transforms.Compose([
            transforms.Resize([256,256]),
            transforms.RandomResizedCrop(256,scale=(0.8,1.2)),
            transforms.RandomRotation(degrees=(-90,90)),
             transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        if kwargs['eval']:
            self.transform = transforms.Compose([
            transforms.Resize([256,256]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.length = 100

        src_root = kwargs['src_root']
        tgt_root = kwargs['tgt_root']
        
        self.src_paths = [os.path.join(src_root,f) for f in os.listdir(src_root) if f.endswith('.png')]
        self.tgt_paths = [os.path.join(tgt_root,f) for f in os.listdir(tgt_root) if f.endswith('.png')]
        self.src_length = len(self.src_paths)
        self.tgt_length = len(self.tgt_paths)
        random.shuffle(self.src_paths)
        random.shuffle(self.tgt_paths)

        self.mx_left_eye_all,\
            self.mn_left_eye_all,\
            self.mx_right_eye_all,\
            self.mn_right_eye_all,\
            self.mx_lip_all,\
            self.mn_lip_all = \
            np.load(kwargs['score_info'])

    def __getitem__(self,i):
        src_idx = i % self.src_length
        tgt_idx = i % self.tgt_length

        src_path = self.src_paths[src_idx]
        tgt_path = self.tgt_paths[tgt_idx]
        exp_path = src_path.replace('img','express')[:-3] + 'npy'

        with Image.open(src_path) as img:
            srcImg = self.transform(img)
        
        with Image.open(tgt_path) as img:
            tgtImg = self.transform(img)

        score = np.load(exp_path)
        score[0] = (score[0] - self.mn_left_eye_all) / (self.mx_left_eye_all - self.mn_left_eye_all)
        score[1] = (score[1] - self.mn_right_eye_all) / (self.mx_right_eye_all - self.mn_right_eye_all)
        score[2] = (score[2] - self.mn_lip_all) / (self.mx_lip_all - self.mn_lip_all)
        score = torch.from_numpy(score.astype(np.float32))

        return srcImg,tgtImg,score


    def __len__(self):
        # return max(self.src_length,self.tgt_length)
        if hasattr(self,'length'):
            return self.length
        else:
            return 10000

