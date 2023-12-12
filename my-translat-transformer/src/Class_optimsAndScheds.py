"""
A class for defining optimizer and schedulers
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
# from src_utils import see_steplr_trend
import cosine_annealing_warmup


class Optims_Scheds():

    def __init__(self, optim="sgd", sched="lin", model=0, cfg=0):
        self.optim = optim.lower()
        self.sched = sched.lower()
        self.model = model
        self.cfg = cfg
        
    def see_steplr_trend(self, step_size=20, num_epochs=200, lr=0.001, final_lr=None, gamma=None, show_plot=False):
        nsteps = int(num_epochs/step_size)
        lr = lr
        if gamma == None and final_lr != None:
            final_lr = final_lr
            gamma = (final_lr/lr)**(1/nsteps)
            print(f"use gamma = {gamma}")

        if gamma != None and final_lr == None:
            final_lr = lr*gamma**nsteps
            print(f"final_lr= {final_lr}")

        y = np.zeros(num_epochs,)
        for i in range(nsteps):
            y[i*step_size:(i+1)*step_size] = lr*gamma**i
        if show_plot:
            plt.yscale("log")  
            plt.plot(y)
            plt.show()

        return gamma, final_lr

    def assign(self):
        O = 0
        S = 0
        if self.optim=="sgd":
            O = torch.optim.SGD(self.model.parameters(), lr=self.cfg.lr, momentum=self.cfg.momentum, weight_decay=self.cfg.wd)
        elif self.optim=="adam":
            O = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.wd)
        elif self.optim=="adamw":
            O = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.wd)
        
        if self.sched=="seqlr_w":
            gamma, _ = self.see_steplr_trend(step_size=20, num_epochs=self.cfg.num_epochs, lr=self.cfg.lr, final_lr=self.cfg.final_lr)
            main_lr_scheduler = torch.optim.lr_scheduler.StepLR(O, step_size=20, gamma=gamma)
    
            warm_up_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=O,
                                                          start_factor=0.033,
                                                          total_iters=3
                                                          )
    
            S = torch.optim.lr_scheduler.SequentialLR(optimizer=O,
                                                      schedulers=[warm_up_scheduler, main_lr_scheduler],
                                                      milestones=[3])    
        
        elif self.sched=="lin":
            S = torch.optim.lr_scheduler.ReduceLROnPlateau(O, mode='min', factor=self.cfg.opt_factor, patience=self.cfg.patience) 
        elif self.sched=="cos":
            S = torch.optim.lr_scheduler.CosineAnnealingLR(O, T_max=10, eta_min=self.cfg.eta_min, last_epoch=self.cfg.last_epoch, verbose=self.cfg.verbose)   
        elif self.sched=="cosw":
            S = cosine_annealing_warmup.CosineAnnealingWarmupRestarts(O, first_cycle_steps=self.cfg.first_cycle_steps, cycle_mult=self.cfg.cycle_mult, max_lr=self.cfg.max_lr, min_lr=self.cfg.min_lr, warmup_steps=self.cfg.warmup_steps, gamma=self.cfg.gamma)
        elif self.sched=="None":
            S = None
        
        self.O = O
        self.S = S

    def optimizer(self):
        return self.O
    
    def scheduler(self):
        return self.S

# END

