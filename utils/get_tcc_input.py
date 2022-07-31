import sys 
sys.path.insert(0,'..')
import pdb 
from model.styleganModule.model import Generator
import torch
from model.styleganModule.utils import *
from utils import convert_img
from model.styleganModule.config import Params as CCNParams
import cv2
import os
import argparse

parser = argparse.ArgumentParser(description="Process")
parser.add_argument('--model_path',default="",type=str,help='')
parser.add_argument('--output_path',default="",type=str,help='')
args = parser.parse_args()

class Process:
    def __init__(self,model_path):
        self.args = CCNParams()
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'

        self.netGt = Generator(
                    self.args.size,self.args.latent,
                    self.args.n_mlp,
                    channel_multiplier=self.args.channel_multiplier).to(self.device)
        self.netGs = Generator(
                    self.args.size,self.args.latent,
                    self.args.n_mlp,
                    channel_multiplier=self.args.channel_multiplier).to(self.device)

        self.loadparams(model_path)

        self.netGs.eval()
        self.netGt.eval()

    def __call__(self,save_base):
        os.makedirs(save_base,exist_ok=True)
        steps = 0
        for i in range(self.args.mx_gen_iters):
            fakes = self.run_sigle()
            for f in fakes:
                cv2.imwrite(os.path.join(save_base,'%06d.png'%steps),f)
                steps += 1
                print('\r have done %06d'%steps,end='',flush=True)

    def run_sigle(self):
        noise = mixing_noise(self.args.infer_batch_size, 
                            self.args.latent, 
                            self.args.mixing,self.device)
        with torch.no_grad():
            latent_s = self.netGs(noise,only_latent=True)
            latent_t = self.netGt(noise,only_latent=True)
            # fake_s,latent_s = self.netGs(noise,return_latents=True)
            # fake_t,latent_t = self.netGt(noise,return_latents=True)
            # mix 
            latent_mix = torch.cat([latent_s[:,:self.args.inject_index],latent_t[:,self.args.inject_index:]],1)
            fake,_ = self.netGt([latent_mix],input_is_latent=True)
            # fake = torch.cat([fake_s,fake_t,fake],-1)
            fake = convert_img(fake,unit=True).permute(0,2,3,1)
            return fake.cpu().numpy()[...,::-1]

    
    def loadparams(self,path):
        ckpt = torch.load(path, map_location=lambda storage, loc: storage)
        self.netGs.load_state_dict(ckpt['Gs'],strict=False)
        self.netGt.load_state_dict(ckpt['gt_ema'],strict=False)
      
if __name__ == "__main__":
    args = parser.parse_args()
    
  
    model = Process(args.model_path)
       
    model(args.output_path)