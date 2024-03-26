import torch
import torchvision
import torchvision.transforms as transforms


DATASET_PATH='CIFAR100/'

transform_train=transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2009, 0.1984, 0.2023))
])
transform_test=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5088, 0.4874, 0.4419), (0.2019, 0.2000, 0.2037))
])

def get_train_data_loader(batch_size):
    train_set=torchvision.datasets.CIFAR100(
    root=DATASET_PATH,
    train=True,
    download=False,
    transform=transform_train    
    )
    return torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True)

def get_test_data_loader(batch_size):
    test_set=torchvision.datasets.CIFAR100(
    root=DATASET_PATH,
    train=False,
    download=False,
    transform=transform_test 
    )
    return torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=True)

def get_mean_and_std(dataset):
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=True)
    mean=torch.zeros(3)
    std=torch.zeros(3)
    print('Computing mean and std')
    for image,label in dataloader:
        for i in range(3):
            mean[i]+=image[:,i,:,:].mean()
            std[i]+=image[:,i,:,:].std()
    mean=mean/len(dataset)
    std=std/len(dataset)
    return mean,std