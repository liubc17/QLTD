import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
# import matplotlib.pyplot as plt

def get_milestone_scheduler(optimizer: torch.optim.Optimizer, step_size, lr_factor):

    milestone_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=lr_factor)

    return milestone_scheduler


def get_cosine_annealing_scheduler(optimizer: torch.optim.Optimizer, T_max: int, eta_min):

    cosine_annealing_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    return cosine_annealing_scheduler


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_min, lr_max, warm_up=0, T_max=10, start_ratio=0.1):
        """
        Description:
            - get warmup consine lr scheduler

        Arguments:
            - optimizer: (torch.optim.*), torch optimizer
            - lr_min: (float), minimum learning rate
            - lr_max: (float), maximum learning rate
            - warm_up: (int),  warm_up epoch or iteration
            - T_max: (int), maximum epoch or iteration
            - start_ratio: (float), to control epoch 0 lr, if ratio=0, then epoch 0 lr is lr_min

        Example:
            <<< epochs = 100
            <<< warm_up = 5
            <<< cosine_lr = WarmupCosineLR(optimizer, 1e-9, 1e-3, warm_up, epochs)
            <<< lrs = []
            <<< for epoch in range(epochs):
            <<<     optimizer.step()
            <<<     lrs.append(optimizer.state_dict()['param_groups'][0]['lr'])
            <<<     cosine_lr.step()
            <<< plt.plot(lrs, color='r')
            <<< plt.show()

        """
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.warm_up = warm_up
        self.T_max = T_max
        self.start_ratio = start_ratio
        self.cur = 0  # current epoch or iteration

        super().__init__(optimizer, -1)

    def get_lr(self):
        if (self.warm_up == 0) & (self.cur == 0):
            lr = self.lr_max
        elif (self.warm_up != 0) & (self.cur <= self.warm_up):
            if self.cur == 0:
                lr = self.lr_min + (self.lr_max - self.lr_min) * (self.cur + self.start_ratio) / self.warm_up
            else:
                lr = self.lr_min + (self.lr_max - self.lr_min) * (self.cur) / self.warm_up
                # print(f'{self.cur} -> {lr}')
        else:
            # this works fine
            lr = self.lr_min + (self.lr_max - self.lr_min) * 0.5 * \
                 (np.cos((self.cur - self.warm_up) / (self.T_max - self.warm_up) * np.pi) + 1)

        self.cur += 1

        return [lr for base_lr in self.base_lrs]


def get_WarmupCosine_scheduler(optimizer: torch.optim.Optimizer, epochs:int, eta_min, eta_max, warm_up_ep:int):

    scheduler = WarmupCosineLR(optimizer, lr_min=eta_min,lr_max=eta_max,warm_up=warm_up_ep,T_max=epochs,start_ratio=0.1)

    return scheduler

# class
# model = torch.load('E:/！！！research/NN_compression/LRQ_ILSVRC2012/save_model/origin_model/Resnet18_pretrained_imagenet.pt')
# optimizer = get_adam_optimizer(model,initial_lr=1e-7)
# epochs = 100
# warm_up = 10
# cosine_lr = WarmupCosineLR(optimizer, 1e-7, 1e-5, warm_up, epochs, 0.1)
# lrs = []
# for epoch in range(epochs):
#     optimizer.step()
#     lrs.append(optimizer.state_dict()['param_groups'][0]['lr'])
#     cosine_lr.step()
#
# plt.figure(figsize=(10, 6))
# plt.plot(lrs, color='r')
# plt.text(0, lrs[0], str(lrs[0]))
# plt.text(epochs, lrs[-1], str(lrs[-1]))
# plt.show()


def get_adam_optimizer(model:torch.nn.Module,initial_lr):

    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)

    return optimizer


def get_SGD_optimizer(model:torch.nn.Module, initial_lr, momentum, weight_decay):

    optimizer = torch.optim.SGD(model.parameters(),lr=initial_lr, momentum =momentum, weight_decay = weight_decay)

    return optimizer