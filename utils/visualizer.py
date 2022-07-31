"""
Copyright (C) 2019 NVIDIA Corporation. ALL rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os 
import time 
import subprocess

from tensorboardX import SummaryWriter



class Visualizer:
    def __init__(self,opt,mode='train'):
        self.opt = opt
        self.name = opt.name 
        self.mode = mode
        self.train_log_dir = os.path.join(opt.checkpoint_path,"logs/%s"%mode)
        self.log_name = os.path.join(opt.checkpoint_path,'loss_log_%s.txt'%mode)
        if opt.local_rank == 0:
            if not os.path.exists(self.train_log_dir):
                os.makedirs(self.train_log_dir)

            self.train_writer = SummaryWriter(self.train_log_dir)

            self.log_file = open(self.log_name,"a")
            now = time.strftime("%c")
            self.log_file.write('================ Training Loss (%s) =================\n'%now)
            self.log_file.flush()


    # errors:dictionary of error labels and values
    def plot_current_errors(self,errors,step):

        for tag,value in errors.items():
            
            self.train_writer.add_scalar("%s/"%self.name+tag,value,step)
            self.train_writer.flush()


    # errors: same format as |errors| of CurrentErrors
    def print_current_errors(self,epoch,i,errors,t):
        message = '(epoch: %d\t iters: %d\t time: %.5f)\t'%(epoch,i,t)
        for k,v in errors.items():

            message += '%s: %.5f\t' %(k,v)

        print(message)

        self.log_file.write('%s\n' % message)
        self.log_file.flush()

    def display_current_results(self, visuals, step):
        if visuals is None:
            return 
        for label, image in visuals.items():
            # Write the image to a string
    
            self.train_writer.add_image("%s/"%self.name+label,image,global_step=step)

    def close(self):
        
        self.train_writer.close()
        self.log_file.close()   