#! /usr/bin/python 
# -*- encoding: utf-8 -*-
'''
@author LeslieZhao
@date 20220721
'''
import torch 

from trainer.ModelTrainer import ModelTrainer
from model.Pix2PixModule.model import Generator,Discriminator,ExpressDetector
from utils.utils import *
from model.Pix2PixModule.module import *
from model.Pix2PixModule.loss import *
import torch.distributed as dist 
import random
import itertools

class TTNTrainer(ModelTrainer):

    def __init__(self, args):
        super().__init__(args)
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        self.netG = Generator(img_channels=3).to(self.device)
        
        self.netTxD = Discriminator(in_channels=1).to(self.device)

        self.netSfD = Discriminator(in_channels=3).to(self.device)

        self.ExpG = None
        if args.use_exp:
            self.ExpG = ExpressDetector().to(self.device)
            self.ExpG.apply(init_weights)

        self.netG.apply(init_weights)
      
        self.netTxD.apply(init_weights)
        self.netSfD.apply(init_weights)

        self.optimG,self.optimD,self.optimExp = self.create_optimizer() 

        if args.pretrain_path is not None:
            self.loadParameters(args.pretrain_path)

        if args.dist:
            self.netG,self.netG_module = self.use_ddp(self.netG)
          
            self.netTxD,self.netTxD_module = self.use_ddp(self.netTxD)
            self.netSfD,self.netSfD_module = self.use_ddp(self.netSfD)

            if args.use_exp:
                self.ExpG,self.ExpG_module = self.use_ddp(self.ExpG)
        else:
            self.netG_module = self.netG
            self.netTxD_module = self.netTxD
            self.netSfD_module = self.netSfD
            if args.use_exp:
                self.ExpG_module = self.ExpG
           

        self.VggLoss = VGGLoss(args.vgg_model).to(self.device).eval()
        self.TVLoss = TVLoss(1).to(self.device).eval()
        self.L1_Loss = nn.L1Loss()
        self.MSE_Loss = nn.MSELoss()
          

    def create_optimizer(self):
        g_optim = torch.optim.Adam(
                    self.netG.parameters(),
                    lr=self.args.lr,
                    betas=(self.args.beta1,self.args.beta2),
                    )
       
        d_optim = torch.optim.Adam(
                    
                    itertools.chain(self.netTxD.parameters(),self.netSfD.parameters()),
                    lr=self.args.lr,
                    betas=(self.args.beta1,self.args.beta2),
                    )
        exp_optim = None 
        if self.args.use_exp:
            exp_optim = torch.optim.Adam(
                        self.ExpG.parameters(),
                        lr=self.args.lr,
                        betas=(self.args.beta1,self.args.beta2),
                        )
     
        return  g_optim,d_optim,exp_optim

    
    def run_single_step(self, data, steps):
        self.netG.train()
        super().run_single_step(data, steps)
        

    def run_discriminator_one_step(self, data,step):
        
        D_losses = {}
        requires_grad(self.netTxD, True)
        requires_grad(self.netSfD, True)
        xs,xt,_ = data 
        with torch.no_grad():
            xg = self.netG(xs)

        # surface
        
        blur_fake = guided_filter(xg,xg,r=5,eps=2e-1)
        blur_style = guided_filter(xt,xt,r=5,eps=2e-1)

        D_blur_real = self.netSfD(blur_style)
        D_blur_fake = self.netSfD(blur_fake) 
        d_loss_surface_real = self.MSE_Loss(D_blur_real,torch.ones_like(D_blur_real))
        d_loss_surface_fake = self.MSE_Loss(D_blur_fake, torch.zeros_like(D_blur_fake))
        d_loss_surface = (d_loss_surface_real + d_loss_surface_fake)/2.0

        D_losses['d_surface_real'] = d_loss_surface_real
        D_losses['d_surface_fake'] = d_loss_surface_fake

        # texture
        gray_fake = color_shift(xg)
        gray_style = color_shift(xt)

        D_gray_real = self.netTxD(gray_style)
        D_gray_fake = self.netTxD(gray_fake.detach())
        d_loss_texture_real = self.MSE_Loss(D_gray_real, torch.ones_like(D_gray_real))
        d_loss_texture_fake = self.MSE_Loss(D_gray_fake, torch.zeros_like(D_gray_fake))
        d_loss_texture = (d_loss_texture_real + d_loss_texture_fake)/2.0

        D_losses['d_texture_real'] = d_loss_texture_real
        D_losses['d_texture_fake'] = d_loss_texture_fake

        d_loss_total = d_loss_surface + d_loss_texture


        self.optimD.zero_grad() 
        d_loss_total.backward()
       
        self.optimD.step()
        self.d_losses = D_losses

    def run_generator_one_step(self, data,step):
        
        G_losses = {}
        requires_grad(self.netG, True)
        requires_grad(self.netTxD, False)
        requires_grad(self.netSfD, False)
        requires_grad(self.ExpG, False)
        xs,xt,exp_gt = data 
            
        G_losses,losses,xg = self.compute_g_loss(xs,exp_gt,step)

        self.netG.zero_grad()
        losses.backward()
        self.optimG.step()

        if self.args.use_exp:
            requires_grad(self.ExpG, True)
            pred_exp = self.ExpG(xg.detach())
            exp_loss = self.MSE_Loss(pred_exp,exp_gt)
            self.optimExp.zero_grad()
            exp_loss.backward()
            self.optimExp.step()
            G_losses['raw_exp_loss'] = exp_loss


        self.g_losses = G_losses
        self.generator = [xs.detach(),xg.detach(),xt.detach()]
        

    def evalution(self,test_loader,steps,epoch):
        
        loss_dict = {}
        counter = 0
        index = random.randint(0,len(test_loader)-1)
        self.netG.eval()
       
        with torch.no_grad():
            for i,data in enumerate(test_loader):
                
                data = self.process_input(data)
                xs,xt,exp_gt = data 
                G_losses,losses,xg = self.compute_g_loss(xs,exp_gt,steps)
                for k,v in G_losses.items():
                    loss_dict[k] = loss_dict.get(k,0) + v.detach()
                if i == index and self.args.rank == 0 :
                    
                    self.val_vis.display_current_results(self.select_img([xs,xg,xt]),steps)
                counter += 1
        
       
        for key,val in loss_dict.items():
            loss_dict[key] /= counter

        if self.args.dist:
            # if self.args.rank == 0 :
            dist_losses = loss_dict.copy()
            for key,val in loss_dict.items():
                
                dist.reduce(dist_losses[key],0)
                value = dist_losses[key].item()
                loss_dict[key] = value / self.args.world_size

        if self.args.rank == 0 :
            self.val_vis.plot_current_errors(loss_dict,steps)
            self.val_vis.print_current_errors(epoch+1,0,loss_dict,0)

        return loss_dict
       

    def compute_g_loss(self,xs,exp_gt,step):
        G_losses = {}

        xg = self.netG(xs)

        # warp_up
        if step < 100:
            lambda_surface = 0
            lambda_texture = 0
            lambda_exp = 0
          
        elif step < 500:
            lambda_surface = self.args.lambda_surface * 0.01
            lambda_texture = self.args.lambda_texture * 0.01
            lambda_exp = self.args.lambda_exp * 0.01
          
        elif step < 1000:
            lambda_surface = self.args.lambda_surface * 0.1
            lambda_texture = self.args.lambda_texture * 0.1
            lambda_exp = self.args.lambda_exp * 0.1
          
        else:
            lambda_surface = self.args.lambda_surface
            lambda_texture = self.args.lambda_texture
            lambda_exp = self.args.lambda_exp
    
      
        # surface
        blur_fake = guided_filter(xg,xg,r=5,eps=2e-1)
        D_blur_fake = self.netSfD(blur_fake)
        g_loss_surface = lambda_surface * self.MSE_Loss(D_blur_fake, torch.ones_like(D_blur_fake))
        G_losses['g_loss_surface'] = g_loss_surface

        # texture
        gray_fake = color_shift(xg)
        D_gray_fake = self.netTxD(gray_fake)
        g_loss_texture = lambda_texture * self.MSE_Loss(D_gray_fake, torch.ones_like(D_gray_fake))
        G_losses['g_loss_texture'] = g_loss_texture

        # content
        content_loss = self.VggLoss(xs,xg) * self.args.lambda_content
        G_losses['content_loss'] = content_loss

        # tv loss
        tv_loss = self.TVLoss(xg) * self.args.lambda_tv
        G_losses['tvloss'] = tv_loss

        # exp loss 
        exp_loss = 0
        if self.args.use_exp:
             exp_pred = self.ExpG(xg)
             exp_loss = self.MSE_Loss(exp_pred,exp_gt) * lambda_exp
             G_losses['exploss'] = exp_loss

        losses = g_loss_surface + g_loss_texture + content_loss + tv_loss + exp_loss * lambda_exp
        G_losses['total_loss'] = losses 
        return G_losses,losses,xg

    def get_latest_losses(self):
        return {**self.g_losses,**self.d_losses}

    def get_latest_generated(self):
        return self.generator

    def get_loss_from_val(self,loss):
        return loss['total_loss']

    def loadParameters(self,path):
        ckpt = torch.load(path, map_location=lambda storage, loc: storage)
        self.netG.load_state_dict(ckpt['netG'],strict=False)
        self.netTxD.load_state_dict(ckpt['netTxD'],strict=False)
        self.netSfD.load_state_dict(ckpt['netSfD'],strict=False)
        self.optimG.load_state_dict(ckpt['g_optim'])
        self.optimD.load_state_dict(ckpt['d_optim'])
        if self.args.use_exp:
            self.ExpG.load_state_dict(ckpt['ExpG'],strict=False)
            self.optimExp.load_state_dict(ckpt['exp_optim'])
    def saveParameters(self,path):
        save_dict = {
                        "netG": self.netG_module.state_dict(),
                        "netTxD": self.netTxD_module.state_dict(),
                        "netSfD": self.netSfD_module.state_dict(),
                        "g_optim": self.optimG.state_dict(),
                        "d_optim": self.optimD.state_dict(),
                        "args": self.args,
                    }
        if self.args.use_exp:
            save_dict['ExpG'] = self.ExpG_module.state_dict()
            save_dict['exp_optim'] = self.optimExp.state_dict()
        torch.save(
                    save_dict,
                    path
                )

    def get_lr(self):
        return self.optimG.state_dict()['param_groups'][0]['lr']
    


    
            


        


    
    

    
