import numpy as np
import os 
import cv2
import torch 
from model.Pix2PixModule.model import Generator
from utils.utils import convert_img

class Infer:
    def __init__(self,model_path):
        self.net = Generator(img_channels=3)
        self.load_checkpoint(model_path)


    def run(self,img):
        if isinstance(img,str):
            img = cv2.imread(img)
        inp = self.preprocess(img)
        with torch.no_grad():
            xg = self.net(inp)
        oup = self.postprocess(xg[0])
        return oup
        
    def load_checkpoint(self,path):
        ckpt = torch.load(path, map_location=lambda storage, loc: storage)
        self.net.load_state_dict(ckpt['netG'],strict=False)
        if torch.cuda.is_available():
            self.net.cuda()
        self.net.eval()

    def preprocess(self,img):
        
        img = (img[...,::-1] / 255.0 - 0.5) * 2
        img = img.transpose(2,0,1)[np.newaxis,:].astype(np.float32)
        img = torch.from_numpy(img)
        if torch.cuda.is_available():
            img = img.cuda()
        return img
    def postprocess(self,img):
        img = convert_img(img,unit=True)
        return img.permute(1,2,0).cpu().numpy()[...,::-1]
        


if __name__ == "__main__":

    path = 'pretrain_models/final.pth'
    model = Infer(path)

    img = cv2.imread('')

    img_h,img_w,_ = img.shape 
    n_h,n_w = img_h // 8 * 8,img_w // 8 * 8
    img = cv2.resize(img,(n_w,n_h))

    oup = model.run(img)[...,::-1]
    cv2.imwrite('output.png',oup)
     

