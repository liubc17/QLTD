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


def train(args, logger):
    # 1.initilazing network
    quantization_model = torch.load(args.load_model_path)
    quantization_model.to(args.device)
    teacher_model = torch.load('save_model/origin_model/Resnet18_pretrained_imagenet.pt')
    teacher_model.to(args.device)
    logger.logger.info('Initializing Network!Good Luck!')
    trainloader = get_trainset(batch_size=args.batch_size)

    # setting optimizer and lr_scheduler
    if args.optimizer_type == 'Adam':
        logger.logger.info('Optimizer_type is Adam')
        optimizer = get_adam_optimizer(quantization_model,initial_lr=args.initial_lr)
    if args.optimizer_type == 'SGD':
        logger.logger.info('Optimizer_type is SGD')
        optimizer = get_SGD_optimizer(quantization_model,initial_lr=args.initial_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.scheduler_type == 'cosine':
        logger.logger.info('Scheduler_type is cosine')
        scheduler = get_cosine_annealing_scheduler(optimizer=optimizer,T_max= len(trainloader) * args.num_epoch, eta_min=args.eta_min)
    if args.scheduler_type == 'milestone':
        logger.logger.info('Scheduler_type is milestone')
        scheduler = get_milestone_scheduler(optimizer=optimizer, step_size=args.step_size,lr_factor=args.lr_factor)
        # scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[150,250,300],gamma=0.1)#每10次训练衰减10倍

    for epoch in range(args.num_epoch):
        train_loss = 0
        train_correct = 0
        for images, labels in trainloader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            optimizer.zero_grad()
            preds = quantization_model(images)
            # calculate student logits
            student_logits = F.log_softmax(preds, dim=1)
            # calculate teacher logits
            with torch.no_grad(): teacher_logits = F.softmax(teacher_model(images), dim=1)
            loss = 0.5 * F.kl_div(student_logits, teacher_logits,reduction='batchmean') + 0.5 * F.cross_entropy(preds, labels)
            # logger.logger.info('Using knowledge distillation.')
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += preds.argmax(dim=1).eq(labels).sum().item()

        present_trainset_acc = train_correct / args.trainset_length

        logger.logger.info(
            'Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch, args.num_epoch, train_loss, present_trainset_acc))
        scheduler.step()
    logger.logger.info('Done training!')
    torch.save(quantization_model, args.save_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--load_model_path', type=str,
                        default='save_model/LRQ/quantization_2_4_k256_compressed_model_imagenet_df15.pt')
    parser.add_argument('--save_model_path', type=str,
                        default='save_model/LRQ/quantization_2_4_k256_compressed_model_imagenet_df_trained_milestone_15ep_Adam_KLCE.pt')
    parser.add_argument('--trainset_length', type=int, default=1281167)
    parser.add_argument('--valset_length', type=int, default=50000)
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--scheduler_type',type=str, default='milestone')
    parser.add_argument('--initial_lr', type=float, default=0.001)
    parser.add_argument('--eta_min',type=float,default=1e-6)
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--lr_factor', type=float, default=0.1)

    parser.add_argument('--optimizer_type',type=str, default='Adam')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay',type=float, default=1e-4)

    opt = parser.parse_args()
    logger = Logger()
    train(opt, logger)