import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from Mydataset import get_train_data_loader,get_test_data_loader
from ResNet18_Baidu import ResNet18
from torch.utils.tensorboard import SummaryWriter


#Preseting parmaters
SAVE_PATH="E:\LearningStuff\DLcode\Pytorch\CIFAR10\Models"
TRAINSET_LENGTH=50000
TESTSET_LENGTH=10000
#Hyperparmeters:
num_epochs=80
learning_rate=0.005
batch_size=256

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'


def train():
    #1.initilazing network
    quantization_model = torch.load('save_decom/QLTD/4_2_0.005_512_R18_C100.pt')
    quantization_model.to(device)
    print("Initializing Network")
    trainloader=get_train_data_loader(batch_size=batch_size)
    optimizer=optim.Adam(quantization_model.parameters(),lr=learning_rate)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
    #4.initilazing tensorboard
    # comment=f'ResNet18 batch_size={batch_size} lr={learning_rate} device={device}'
    # tb=SummaryWriter(comment=comment)

    for epoch in range(num_epochs):
        train_loss=0
        train_correct=0
        for images,labels in trainloader:
            images=images.to(device)
            labels=labels.to(device)
            optimizer.zero_grad()
            preds=quantization_model(images)
            loss=F.cross_entropy(preds,labels)
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
            train_correct+=preds.argmax(dim=1).eq(labels).sum().item()

        present_trainset_acc=train_correct/TRAINSET_LENGTH

        print('Train Loss',train_loss,epoch)
        print('Accuracy on Trainset',present_trainset_acc,epoch)
        scheduler.step()

    ckpt = 'save_fine/QLTD/4_2_0.005_512_R18_C100.pt'
    torch.save(quantization_model, ckpt)
    print(ckpt)


if __name__=='__main__':
    train()
