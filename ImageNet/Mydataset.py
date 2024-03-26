import torch
import torchvision
from torchvision.models.resnet import model_urls
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn as nn


Train_path = "I:\ILSRVC2012\Train"
Val_path = "I:\ILSRVC2012\Val"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STDEV = [0.229, 0.224, 0.225]

IMAGENET_VAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STDEV),
    ]
)

IMAGENET_TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STDEV),
    ]
)

def get_trainset(batch_size):
    train_datasets = datasets.ImageFolder(Train_path, transform=IMAGENET_TRAIN_TRANSFORM)
    # print('num of train', len(train_datasets))
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True,num_workers=8,pin_memory=True)

    return train_dataloader

def get_valset(batch_size):
    val_datasets = datasets.ImageFolder(Val_path, transform=IMAGENET_VAL_TRANSFORM)
    # print('num of test', len(val_datasets))
    val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=True, num_workers=8,pin_memory=True)

    return val_dataloader
