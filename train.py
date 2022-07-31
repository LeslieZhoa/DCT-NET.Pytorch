'''
@author LeslieZhao
@date 20220721
'''
import os 

import argparse
from trainer.CCNTrainer import CCNTrainer 
 
from trainer.TTNTrainer import TTNTrainer
import torch.distributed as dist 
from utils.utils import setup_seed,get_data_loader,merge_args
from model.styleganModule.config import Params as CCNParams
from model.Pix2PixModule.config import Params as TTNParams

# torch.multiprocessing.set_start_method('spawn')

parser = argparse.ArgumentParser(description="StyleGAN")
#---------train set-------------------------------------
parser.add_argument('--model',default="ccn",help='')
parser.add_argument('--isTrain',action="store_false",help='')
parser.add_argument('--dist',action="store_false",help='')
parser.add_argument('--batch_size',default=16,type=int)
parser.add_argument('--seed',default=10,type=int)
parser.add_argument('--eval',default=1,type=int,help='whether use eval')
parser.add_argument('--nDataLoaderThread',default=5,type=int,help='Num of loader threads')
parser.add_argument('--print_interval',default=100,type=int)
parser.add_argument('--test_interval',default=100,type=int,help='Test and save every [test_intervaal] epochs')
parser.add_argument('--save_interval',default=100,type=int,help='save model interval')
parser.add_argument('--stop_interval',default=20,type=int)
parser.add_argument('--begin_it',default=0,type=int,help='begin epoch')
parser.add_argument('--mx_data_length',default=100,type=int,help='max data length')
parser.add_argument('--max_epoch',default=10000,type=int)
parser.add_argument('--early_stop',action="store_true",help='')
parser.add_argument('--scratch',action="store_true",help='')
#---------path set--------------------------------------
parser.add_argument('--checkpoint_path',default='checkpoint-onlybaby',type=str)
parser.add_argument('--pretrain_path',default=None,type=str)

# ------optimizer set--------------------------------------
parser.add_argument('--lr',default=0.002,type=float,help="Learning rate")

parser.add_argument(
            '--local_rank',
            type=int,
            default=0,
            help='Local rank passed from distributed launcher'
)

args = parser.parse_args()

def train_net(args):
    train_loader,test_loader,mx_length = get_data_loader(args)
        
    args.mx_data_length = mx_length
    if args.model == 'ccn':
        trainer = CCNTrainer(args)
    if args.model == 'ttn':
        trainer = TTNTrainer(args)

    trainer.train_network(train_loader,test_loader)

if __name__ == "__main__":
    
    args = parser.parse_args()
    
    if args.model == 'ccn':
        params = CCNParams()
    if args.model == 'ttn':
        params = TTNParams()
    
    args = merge_args(args,params)
    if args.dist:
        dist.init_process_group(backend="nccl") # backbend='nccl'
        dist.barrier() # 用于同步训练
        args.world_size = dist.get_world_size() # 一共有几个节点
        args.rank = dist.get_rank() # 当前节点编号
        
    else:
        args.world_size = 1
        args.rank = 0

    setup_seed(args.seed+args.rank)
    print(args)

    args.checkpoint_path = os.path.join(args.checkpoint_path,args.name)
  
    print("local_rank %d | rank %d | world_size: %d"%(int(os.environ.get('LOCAL_RANK','0')),args.rank,args.world_size))
    if args.rank == 0 :
        if not os.path.exists(args.checkpoint_path):
            os.makedirs(args.checkpoint_path)
            print("make dir: ",args.checkpoint_path)
    train_net(args)


    
