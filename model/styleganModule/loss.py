from torch.nn import functional as F
from torch import autograd
from torch import nn
import torch
from .arcface import Backbone
import math
from .op import conv2d_gradfix

class IDLoss(nn.Module):
    def __init__(self,pretrain_model, requires_grad=False):
        super(IDLoss, self).__init__()
        self.idModel = Backbone(50,0.6,'ir_se')
        self.idModel.load_state_dict(torch.load(pretrain_model),strict=False)
        self.idModel.eval()
        self.criterion = nn.CosineSimilarity(dim=1,eps=1e-6)
        self.id_size = 112
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        

    def forward(self, x, y):
        x_id, _ = self.idModel(F.interpolate(x[:,:,28:228,28:228],[self.id_size, self.id_size], mode='bilinear'))
        y_id,_ = self.idModel(F.interpolate(y[:,:,28:228,28:228], 
                            [self.id_size, self.id_size], mode='bilinear'))
        loss = 1 - self.criterion(x_id,y_id)
        return loss.mean()


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths

def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()