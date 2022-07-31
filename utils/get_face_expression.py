import sys 

import pdb
import cv2
import os
from multiprocessing import Pool
import time
import math
import multiprocessing as mp
import numpy as np

import argparse

parser = argparse.ArgumentParser(description="Process")
parser.add_argument('--img_base',default="",type=str,help='')
parser.add_argument('--pool_num',default=2,type=int,help='')
parser.add_argument('--LVT',default='',type=str,help='')
parser.add_argument('--train',action="store_true",help='')
args = parser.parse_args()

class Process:
    def __init__(self):
        self.engine = Engine(
                    face_lmk_path='')
    
    def run(self,img_paths):
        mx_left_eye = -1
        mn_left_eye = 100

        mx_right_eye = -1
        mn_right_eye = 100

        mx_lip = -1
        mn_lip = 100
        for i,img_path in enumerate(img_paths):
            img = cv2.imread(img_path)
            left_eye_score,right_eye_score,lip_score = \
                self.run_single(img)

            base,img_name = os.path.split(img_path)
            score_base = base.replace('img','express')
            os.makedirs(score_base,exist_ok=True)
            np.save(os.path.join(score_base,img_name.split('.')[0]+'.npy'),[left_eye_score,right_eye_score,lip_score])

            mx_left_eye = max(left_eye_score,mx_left_eye)
            mn_left_eye = min(left_eye_score,mn_left_eye)

            mx_right_eye = max(right_eye_score,mx_right_eye)
            mn_right_eye = min(right_eye_score,mn_right_eye)

            mx_lip = max(lip_score,mx_lip)
            mn_lip = min(lip_score,mn_lip)
            print('\rhave done %04d'%i,end='',flush=True)
        print()
        return mx_left_eye,mn_left_eye,mx_right_eye,mn_right_eye,mx_lip,mn_lip


    def run_single(self,img):
        inp = self.engine.preprocess_lmk(img)
        lmk = self.engine.get_lmk(inp)
        lmk = self.engine.postprocess_lmk(lmk,256,[0,0])
        scores = self.get_expression(lmk[0])
        return scores
    def get_expression(self,lmk):
        left_eye_h = abs(lmk[66,1]-lmk[62,1])
        left_eye_w = abs(lmk[60,0]-lmk[64,0])
        left_eye_score = left_eye_h / max(left_eye_w,1e-5)

        right_eye_h = abs(lmk[70,1]-lmk[74,1])
        right_eye_w = abs(lmk[68,0]-lmk[72,0])
        right_eye_score = right_eye_h / max(right_eye_w,1e-5)

        lip_h = abs(lmk[90,1]-lmk[94,1])
        lip_w = abs(lmk[88,0]-lmk[82,0])
        lip_score = lip_h / max(lip_w,1e-5)

        return left_eye_score,right_eye_score,lip_score

def work(queue,img_paths):
    model = Process()
    mx_left_eye,mn_left_eye,mx_right_eye,mn_right_eye,mx_lip,mn_lip = \
        model.run(img_paths)
    queue.put([mx_left_eye,mn_left_eye,mx_right_eye,mn_right_eye,mx_lip,mn_lip])

def print_error(value):
    print("error: ", value)
if __name__ == "__main__":

    args = parser.parse_args()
    # import LVT base path
    sys.path.insert(0,args.LVT)
    from LVT import Engine
    from utils import utils

    mp.set_start_method('spawn')
    m = mp.Manager()
    queue = m.Queue()
    model = Process()
    base = args.img_base
    img_paths = [os.path.join(base,f) for f in os.listdir(base)]
    pool_num = args.pool_num
    length = len(img_paths)
   
    dis = math.ceil(length/float(pool_num))
   
   
    t1 = time.time()
    print('***************all length: %d ******************'%length)
    p = Pool(pool_num)
    for i in range(pool_num):
        p.apply_async(work, args = (queue,img_paths[i*dis:(i+1)*dis],),error_callback=print_error)

    p.close()
    p.join()
    print("all the time: %s"%(time.time()-t1))

    if args.train:
        mx_left_eye_all = -1
        mn_left_eye_all = 100

        mx_right_eye_all = -1
        mn_right_eye_all = 100

        mx_lip_all = -1
        mn_lip_all = 100
        
        while not queue.empty():
            mx_left_eye,mn_left_eye,mx_right_eye,mn_right_eye,mx_lip,mn_lip = \
                queue.get()

            mx_left_eye_all = max(mx_left_eye_all,mx_left_eye)
            mn_left_eye_all = min(mn_left_eye_all,mn_left_eye)

            mx_right_eye_all = max(mx_right_eye_all,mx_right_eye)
            mn_right_eye_all = min(mn_right_eye_all,mn_right_eye)

            mx_lip_all = max(mx_lip_all,mx_lip)
            mn_lip_all = min(mn_lip_all,mn_lip)
        os.makedirs('../pretrain_models',exist_ok=True)
        np.save('../pretrain_models/all_express_mean.npy',[mx_left_eye_all,mn_left_eye_all,
                                        mx_right_eye_all,mn_right_eye_all,
                                        mx_lip_all,mn_lip_all])
        
