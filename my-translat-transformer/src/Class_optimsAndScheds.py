"""
A class for defining optimizer and schedulers
"""

import torch
from torch import nn

import cosine_annealing_warmup


class Optims_Scheds():

    def __init__(self, optim="sgd", sched="lin", model=0, cfg=0):
        self.optim = optim.lower()
        self.sched = sched.lower()
        self.model = model
        self.cfg = cfg

    def assign(self):
        O = 0
        S = 0
        if self.optim=="sgd":
            O = torch.optim.SGD(self.model.parameters(), lr=self.cfg.lr, momentum=self.cfg.momentum, weight_decay=self.cfg.wd)
        elif self.optim=="adam":
            O = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.wd)
        elif self.optim=="adamw":
            O = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.wd)

        if self.sched=="lin":
            S = torch.optim.lr_scheduler.ReduceLROnPlateau(O, mode='min', factor=self.cfg.opt_factor, patience=self.cfg.patience) 
        elif self.sched=="cos":
            S = torch.optim.lr_scheduler.CosineAnnealingLR(O, T_max=10, eta_min=self.cfg.eta_min, last_epoch=self.cfg.last_epoch, verbose=self.cfg.verbose)   
        elif self.sched=="cosw":
            S = cosine_annealing_warmup.CosineAnnealingWarmupRestarts(O, first_cycle_steps=self.cfg.first_cycle_steps, cycle_mult=self.cfg.cycle_mult, max_lr=self.cfg.max_lr, min_lr=self.cfg.min_lr, warmup_steps=self.cfg.warmup_steps, gamma=self.cfg.gamma)
        
        self.O = O
        self.S = S

    def optimizer(self):
        return self.O
    
    def scheduler(self):
        return self.S

# END

