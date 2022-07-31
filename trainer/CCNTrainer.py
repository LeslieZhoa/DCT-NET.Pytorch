#! /usr/bin/python 
# -*- encoding: utf-8 -*-
'''
@author LeslieZhao
@date 20220721
'''
import torch 

from trainer.ModelTrainer import ModelTrainer
from model.styleganModule.model import Generator,Discriminator
from model.styleganModule.utils import *
from model.styleganModule.loss import *
from utils.utils import *


class CCNTrainer(ModelTrainer):

    def __init__(self, args):
        super().__init__(args)
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
      
        self.netGs = Generator(
                    args.size,args.latent,
                    args.n_mlp,
                    channel_multiplier=args.channel_multiplier).to(self.device)

        self.netGt = Generator(args.size,args.latent,
                    args.n_mlp,
                    channel_multiplier=args.channel_multiplier).to(self.device)
        
        
        self.netD = Discriminator(
                args.size, channel_multiplier=args.channel_multiplier).to(self.device)
        self.gt_ema = Generator(
                args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(self.device)
        self.gt_ema.eval()
        accumulate(self.gt_ema, self.netGt, 0)
        self.g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
        self.d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

        self.ckpt = None
            # self.init_weights(init_type='kaiming')
        if not args.scratch:
            ckpt = torch.load(args.stylegan_path, map_location=lambda storage, loc: storage)
            self.netGs.load_state_dict(ckpt['g_ema'],strict=False)
            self.netGt.load_state_dict(ckpt['g'],strict=False)
            self.gt_ema.load_state_dict(ckpt['g_ema'],strict=False)
            self.netD.load_state_dict(ckpt["d"])
            self.ckpt = ckpt
        self.optimG,self.optimD = self.create_optimizer() 
        
        self.sample_z = torch.randn(args.n_sample, args.latent).to(self.device)


        if args.pretrain_path is not None:
            self.loadParameters(args.pretrain_path)

        if args.dist:
            self.netGt,self.netGt_module = self.use_ddp(self.netGt)
            self.netD,self.netD_module = self.use_ddp(self.netD)
        else:
            self.netGt_module = self.netGt 
            self.netD_module = self.netD
        self.netGs.eval()
        
        self.accum = 0.5 ** (32 / (10 * 1000))
        self.criterionID = IDLoss(args.id_model).to(self.device)
            
        self.mean_path_length = 0
        

    def create_optimizer(self):
        g_optim = torch.optim.Adam(
                    self.netGt.parameters(),
                    lr=self.args.lr * self.g_reg_ratio,
                    betas=(0 ** self.g_reg_ratio, 0.99 ** self.g_reg_ratio),
                    )
        d_optim = torch.optim.Adam(
                    self.netD.parameters(),
                    lr=self.args.lr * self.d_reg_ratio,
                    betas=(0 ** self.d_reg_ratio, 0.99 ** self.d_reg_ratio),
                    )
        if self.ckpt is not None:
            g_optim.load_state_dict(self.ckpt["g_optim"])
            d_optim.load_state_dict(self.ckpt["d_optim"])
        return  g_optim,d_optim

    
    def run_single_step(self, data, steps):
        self.netGt.train()
        if self.args.interval_train:
            data = self.process_input(data)
            if not hasattr(self,'interval_flag'):
                self.interval_flag = True 
            if self.interval_flag:
                self.run_generator_one_step(data,steps)
                self.d_losses = {}
            else:
                self.run_discriminator_one_step(data,steps)
                self.g_losses = {}
            if steps % self.args.interval_steps == 0:
                self.interval_flag = not self.interval_flag
            
            
        else:
            super().run_single_step(data, steps)
        

    def run_discriminator_one_step(self, data,step):
        
        D_losses = {}
        requires_grad(self.netGt, False)
        requires_grad(self.netD, True)
        noise = mixing_noise(self.args.batch_size, 
                            self.args.latent, 
                            self.args.mixing,self.device)

        fake_img, _ = self.netGt(noise)
        fake_pred = self.netD(fake_img)
        real_pred = self.netD(data)
        d_loss = d_logistic_loss(real_pred, fake_pred)
        D_losses['d'] = d_loss
        
        self.netD.zero_grad()
        d_loss.backward()
        self.optimD.step()

        if step % self.args.d_reg_every == 0:
            data.requires_grad = True 
            
            real_pred = self.netD(data)
            r1_loss = d_r1_loss(real_pred,data)
            self.netD.zero_grad()
            r1_loss = self.args.r1 / 2 * \
                      r1_loss * self.args.d_reg_every + \
                      0 * real_pred[0]
            
            r1_loss.mean().backward()
          
            self.optimD.step()
            D_losses['r1'] = r1_loss
        
        self.d_losses = D_losses


    def run_generator_one_step(self, data,step):
        
        G_losses = {}
        requires_grad(self.netGt, True)
        requires_grad(self.netD, False)
        requires_grad(self.netGs, False)
        
        noise = mixing_noise(self.args.batch_size, 
                            self.args.latent, 
                            self.args.mixing,self.device)

        fake_s,_ = self.netGs(noise)
        fake_t,_ = self.netGt(noise)
        fake_pred = self.netD(fake_t)
        gan_loss = g_nonsaturating_loss(fake_pred) * self.args.lambda_gan
        id_loss =  self.criterionID(fake_s,fake_t) * self.args.lambda_id
        G_losses['g'] = gan_loss
        G_losses['id'] = id_loss
        losses = gan_loss + id_loss 
        G_losses['g_losses'] = losses
        self.netGt.zero_grad()
        losses.mean().backward()
        self.optimG.step()
        
        if step % self.args.g_reg_every == 0:
            path_batch_size = max(1, self.args.batch_size // self.args.path_batch_shrink)
            noise = mixing_noise(path_batch_size, self.args.latent, self.args.mixing,self.device)
            fake_img, latents = self.netGt(noise, return_latents=True)

            path_loss, self.mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, self.mean_path_length
            )

            weighted_path_loss = self.args.path_regularize * self.args.g_reg_every * path_loss
            if self.args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            self.netGt.zero_grad()
            weighted_path_loss.mean().backward()
            self.optimG.step()

            G_losses['path'] = weighted_path_loss
        
        accumulate(self.gt_ema,self.netGt_module,self.accum)
        self.g_losses = G_losses
        self.generator = [data.detach(),fake_s.detach(),fake_t.detach()]
        
    
    def evalution(self,test_loader,steps,epoch):
        
        loss_dict = {}
        with torch.no_grad():
            fake_s,_ = self.netGs([self.sample_z])
            fake_t,_ = self.gt_ema([self.sample_z])
        if  self.args.rank == 0 :
                self.val_vis.display_current_results(self.select_img([fake_s,fake_t]),steps)
                #  self.val_vis.display_current_results(self.select_img([fake_t]),steps)
        
        return loss_dict

    def get_latest_losses(self):
        return {**self.g_losses,**self.d_losses}

    def get_latest_generated(self):
        return self.generator

    def loadParameters(self,path):
        ckpt = torch.load(path, map_location=lambda storage, loc: storage)
        self.netGs.load_state_dict(ckpt['Gs'],strict=False)
        self.netGt.load_state_dict(ckpt['Gt'],strict=False)
        self.gt_ema.load_state_dict(ckpt['gt_ema'],strict=False)
        self.optimG.load_state_dict(ckpt['g_optim'])
        self.optimD.load_state_dict(ckpt['d_optim'])

    def saveParameters(self,path):
        torch.save(
                    {
                        "Gs": self.netGs.state_dict(),
                        "Gt": self.netGt_module.state_dict(),
                        "gt_ema": self.gt_ema.state_dict(),
                        "g_optim": self.optimG.state_dict(),
                        "d_optim": self.optimD.state_dict(),
                        "args": self.args,
                    },
                    path
                )

    def get_lr(self):
        return self.optimG.state_dict()['param_groups'][0]['lr']
    


    
            


        


    
    

    
