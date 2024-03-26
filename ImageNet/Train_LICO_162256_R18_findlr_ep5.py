import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Mydataset_LICO import get_trainset, get_valset
from Resnet18_torchvision import ResNet18
# from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from Logger import Logger
from lr_scheduler_utils.lr_scheduler import get_adam_optimizer,get_SGD_optimizer,get_milestone_scheduler,get_cosine_annealing_scheduler
from lr_scheduler_utils.cyclic_scheduler import CyclicLRWithRestarts
import math

def train(args, logger):
    # 1.initilazing network
    quantization_model = torch.load(args.load_model_path)
    quantization_model.to(args.device)
    ckpt = torch.load('Find_lr_utils/Artificial_lr/quantization_1.6_2_k256_compressed_model_imagenet_df30_artificial_ep4.pth')
    quantization_model.load_state_dict(ckpt['net'])
    logger.logger.info(args.save_model_path)
    logger.logger.info('Initializing Network!Good Luck!')
    trainloader = get_trainset(batch_size=args.batch_size)

    # setting optimizer and lr_scheduler
    if args.optimizer_type == 'Adam':
        logger.logger.info('Optimizer_type is Adam')
        optimizer = get_adam_optimizer(quantization_model,initial_lr=args.initial_lr)
    if args.optimizer_type == 'SGD':
        logger.logger.info('Optimizer_type is SGD')
        optimizer = get_SGD_optimizer(quantization_model,initial_lr=args.initial_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    optimizer.load_state_dict(ckpt['optimizer'])

    if args.scheduler_type == 'cosine':
        logger.logger.info('Scheduler_type is cosine')
        scheduler = get_cosine_annealing_scheduler(optimizer=optimizer,T_max= len(trainloader) * args.num_epoch, eta_min=args.eta_min)
    if args.scheduler_type == 'milestone':
        logger.logger.info('Scheduler_type is milestone')
        scheduler = get_milestone_scheduler(optimizer=optimizer, step_size=args.step_size,lr_factor=args.lr_factor)
        # scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[150,250,300],gamma=0.1)#每10次训练衰减10倍
    if args.scheduler_type == 'cyclic':
        logger.logger.info('Scheduler_type is cyclic')
        # scheduler = CyclicLRWithRestarts(optimizer, args.batch_size, args.trainset_length, restart_period=5, t_mult=1.2,
        #                                  policy="triangular")
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-8, max_lr=1e-2, step_size_up=30, step_size_down=30, mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)

    if args.scheduler_type == 'Plateau':
        logger.logger.info('Scheduler_type is Plateau')
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.1,patience=3,min_lr=0,eps=1e-8,threshold_mode='rel')

    init_value = 1e-8
    final_value = 10.
    beta = 0.98
    train_length = len(trainloader) - 1
    lr_factor = (final_value / init_value) ** (1/train_length)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []

    for batch_id,(images, labels) in enumerate(trainloader):
        batch_num += 1
        images = images.to(args.device)
        labels = labels.to(args.device)
        optimizer.zero_grad()
        preds = quantization_model(images)
        loss = F.cross_entropy(preds, labels)
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** batch_num)
        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        # Record the best loss
        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss
        # Store the values
        logger.logger.info('losses:{:.3f}'.format(smoothed_loss))
        logger.logger.info('log_lrs:{}'.format(math.log10(lr)))
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        # logger.logger.info('I am training! !')
        loss.backward()
        optimizer.step()
        lr *= lr_factor
        optimizer.param_groups[0]['lr'] = lr

    # 将losses和log_lrs写入txt
    for id, item in enumerate(losses):
        losses[id] = str(item)
    f1 = open("losses1_R18_162256.txt","w")
    n = '\n'
    f1.write(n.join(losses))
    f1.close()

    for id,item in enumerate(log_lrs):
        log_lrs[id] = str(item)
    f2 = open("log_lrs1_R18_162256.txt","w")
    f2.write(n.join(log_lrs))
    f2.close()

def test(quantization_model,args):
    test_loader=get_valset(128)
    quantization_model.to(args.device)
    quantization_model.eval()
    total_correct=0
    total_loss=0
    with torch.no_grad():
        for images,labels in test_loader:
            images=images.to(args.device)
            labels=labels.to(args.device)
            preds=quantization_model(images)
            loss=F.cross_entropy(preds,labels)
            total_loss+=loss.item()
            total_correct+=preds.argmax(dim=1).eq(labels).sum().item()
        test_acc=total_correct/args.valset_length
    return test_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--load_model_path', type=str,
                        default='save_model/R18/LRQ/quantization_1.6_2_k256_compressed_model_imagenet_df30.pt')
    parser.add_argument('--save_model_path', type=str,
                        default='save_model/R18/LRQ/quantization_2_4_k256_compressed_model_imagenet_df15_torchcyclic_trained.pt')
    parser.add_argument('--trainset_length', type=int, default=1281167)
    parser.add_argument('--valset_length', type=int, default=50000)
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--scheduler_type',type=str, default='cyclic')
    parser.add_argument('--initial_lr', type=float, default=1e-5)
    parser.add_argument('--eta_min',type=float,default=1e-6)
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--lr_factor', type=float, default=0.1)

    parser.add_argument('--optimizer_type',type=str, default='Adam')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay',type=float, default=1e-4)

    opt = parser.parse_args()
    logger = Logger()
    train(opt, logger)