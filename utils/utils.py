
'''
@author LeslieZhao
@date 20220721
'''
import torch 
from data.CCNLoader import CCNData
from data.TTNLoader import TTNData

import os
import torch.distributed as dist

def reduce_loss_dict(loss_dict):
    world_size = int(os.environ.get("WORLD_SIZE","1"))

    if world_size < 2:
        return loss_dict

    with torch.no_grad():
        keys = []
        losses = []

        for k in sorted(loss_dict.keys()):
            keys.append(k)
            losses.append(loss_dict[k])

        losses = torch.stack(losses, 0)
        dist.reduce(losses, dst=0)

        if dist.get_rank() == 0:
            losses /= world_size

        reduced_losses = {k: v for k, v in zip(keys, losses)}

    return reduced_losses

def requires_grad(model, flag=True):
    if model is None:
        return 
    for p in model.parameters():
        p.requires_grad = flag
def need_grad(x):
    x = x.detach()
    x.requires_grad_()
    return x

def init_weights(model,init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                torch.nn.init.normal_(m.weight.data, 1.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)

        elif hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
                
        elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'xavier_uniform':
                torch.nn.init.xavier_uniform_(m.weight.data, gain=1.0)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'none':  # uses pytorch's default init method
                m.reset_parameters()
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            
    model.apply(init_func)

def accumulate(model1, model2, decay=0.999,use_buffer=False):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)
   
    if use_buffer:
        for p1,p2 in zip(model1.buffers(),model2.buffers()):
            p1.detach().copy_(decay*p1.detach()+(1-decay)*p2.detach())
            
def setup_seed(seed):
     torch.manual_seed(seed)
     if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def get_data_loader(args):
    if args.model == 'ccn':
        train_data = CCNData(root=args.root,dist=args.dist)
        test_data = None 
    if args.model == 'ttn':
        train_data = TTNData(dist=args.dist,eval=False,
                            src_root=args.train_src_root,
                            tgt_root=args.train_tgt_root,
                            score_info=args.score_info)
        test_data = TTNData(dist=args.dist,eval=True,
                            src_root=args.val_src_root,
                            tgt_root=args.val_tgt_root,
                            score_info=args.score_info)

    train_loader = torch.utils.data.DataLoader(
                        train_data,
                        batch_size=args.batch_size,
                        num_workers=args.nDataLoaderThread,
                        pin_memory=False,
                        drop_last=True
                    )
    test_loader = None if test_data is None else \
        torch.utils.data.DataLoader(
                        test_data,
                        batch_size=args.batch_size,
                        num_workers=args.nDataLoaderThread,
                        pin_memory=False,
                        drop_last=True
                    )
    return train_loader,test_loader,len(train_data) 



def merge_args(args,params):
   for k,v in vars(params).items():
      setattr(args,k,v)
   return args

def convert_img(img,unit=False):
   
    img = (img + 1) * 0.5
    if unit:
        return torch.clamp(img*255+0.5,0,255)
    
    return torch.clamp(img,0,1)