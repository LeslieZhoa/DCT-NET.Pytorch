import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *

from torch.autograd import Variable
# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self,model_path):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19(in_channels=3,
                    VGGtype='VGG19',
                    init_weights=model_path,
                    batch_norm=False, feature_mode=True)
        self.vgg.eval()
        self.criterion = nn.L1Loss()
        

    def forward(self, x, y):
        
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        _, c, h, w = x_vgg.shape
        loss = self.criterion(x_vgg,y_vgg)
        return loss *255 / (c*h*w)


    

class TVLoss(nn.Module):
    def __init__(self, k_size):
        super().__init__()
        self.k_size = k_size

    def forward(self, image):
        b, c, h, w = image.shape
        tv_h = torch.mean((image[:, :, self.k_size:, :] - image[:, :, : -self.k_size, :])**2)
        tv_w = torch.mean((image[:, :, :, self.k_size:] - image[:, :, :, : -self.k_size])**2)
        tv_loss = (tv_h + tv_w) / (3 * h * w)
        return tv_loss.mean()


