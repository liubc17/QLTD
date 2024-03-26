import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from Mydataset import get_trainset, get_valset
from Resnet18_torchvision import ResNet18
# from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from Logger import Logger
from lr_scheduler_utils.cyclic_scheduler import CyclicLRWithRestarts



def train(args,logger):
    #1.initilazing network
    quantization_model = torch.load(args.load_model_path)
    quantization_model.to(args.device)
    logger.logger.info('Initializing Network!Good Luck!')
    trainloader = get_trainset(batch_size=args.batch_size)
    optimizer=optim.Adam(quantization_model.parameters(),lr=args.learning_rate)
    # scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[150,250,300],gamma=0.1)#每10次训练衰减10倍
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = CyclicLRWithRestarts(optimizer, args.batch_size, args.trainset_length, restart_period=5, t_mult=1.2, policy="triangular")
    for epoch in range(args.num_epoch):
        train_loss=0
        train_correct=0
        for images,labels in trainloader:
            images=images.to(args.device)
            labels=labels.to(args.device)
            optimizer.zero_grad()
            preds=quantization_model(images)
            loss=F.cross_entropy(preds,labels)
            logger.logger.info('I am training! !')
            loss.backward()
            optimizer.step()
            scheduler.batch_step()
            train_loss+=loss.item()
            train_correct+=preds.argmax(dim=1).eq(labels).sum().item()
            
        
        present_trainset_acc=train_correct/args.trainset_length

        logger.logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch, args.num_epoch, train_loss, present_trainset_acc))
        scheduler.step()
    logger.logger.info('Done training!')
    torch.save(quantization_model, args.save_model_path)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch',type=int,default=30)
    parser.add_argument('--learning_rate',type=float,default=0.001)
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--load_model_path',type=str,default='save_model/R50/lowrank/lowrank_2_compressed_model_imagenet.pt')
    parser.add_argument('--save_model_path',type=str,default='save_model/R50/lowrank/lowrank_2_compressed_model_imagenet_15ep_trained.pt')
    parser.add_argument('--trainset_length',type=int,default=1281167)
    parser.add_argument('--valset_length',type=int,default=50000)
    parser.add_argument('--device',type=str,default='cuda')
    opt = parser.parse_args()
    logger = Logger()
    train(opt,logger)