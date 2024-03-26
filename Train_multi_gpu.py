import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Mydataset_multi_gpu import get_train_data_loader, get_test_data_loader
from ResNet18_Baidu import ResNet18
import os

# from torch.utils.tensorboard import SummaryWriter

# Preseting parmaters
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'
TRAINSET_LENGTH = 50000
TESTSET_LENGTH = 10000
# Hyperparmeters:
num_epochs = 90
learning_rate = 0.001
batch_size = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_gpu = torch.cuda.device_count()
device_ids = list(range(num_gpu))


def train():
    # 1.initilazing network
    torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
    quantization_model = torch.load(
        'save_model/quantization_1.6_2_k2048_compressed_model_cifar10_nozip.pt')
    quantization_model = quantization_model.to(device)
    quantization_model = torch.nn.parallel.DistributedDataParallel(quantization_model)
    print("Initializing Network")
    trainloader = get_train_data_loader(batch_size=batch_size)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainloader)
    optimizer = optim.Adam(quantization_model.parameters(), lr=learning_rate)
    # scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[150,250,300],gamma=0.1)#每10次训练衰减10倍
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        train_loss = 0
        train_correct = 0
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            preds = quantization_model(images)
            loss = F.cross_entropy(preds, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += preds.argmax(dim=1).eq(labels).sum().item()

        present_trainset_acc = train_correct / TRAINSET_LENGTH

        print('Train Loss', train_loss, epoch)
        print('Accuracy on Trainset', present_trainset_acc, epoch)
        scheduler.step()

    torch.save(quantization_model,
               'save_model/LRQ/parameters/centroids/quantization_2_4_k2048_compressed_model_cifar10_trained.pt')


if __name__ == '__main__':
    train()