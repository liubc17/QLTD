import torch
import torchvision
from torchvision.models.resnet import model_urls
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn as nn

def get_resnet18(pretrained: bool = False) -> torch.nn.Module:
    """Get PyTorch's default ResNet-18 model"""
    # Hack to fix SSL error while loading pretrained model -- see https://github.com/pytorch/pytorch/issues/2271
    model_urls["resnet18"] = model_urls["resnet18"].replace("https://", "http://")
    model = torchvision.models.resnet18(pretrained=pretrained)
    model._arch = "resnet18"
    return model

def val(model,val_dataloader):
    model.eval()
    model = model.cuda()
    eval_loss= 0.
    eval_acc = 0.
    for batch_x, batch_y in val_dataloader:
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        out = model(batch_x)
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(out, batch_y)
        eval_loss += loss.item()
        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum()
        eval_acc += num_correct.item()
        print(1)
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(val_datasets)), eval_acc / (len(val_datasets))))
if __name__ == '__main__':
    model = get_resnet18(True)


    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STDEV = [0.229, 0.224, 0.225]

    IMAGENET_TRAIN_TRANSFORM = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STDEV),
        ]
    )
    batch_size = 128

    val_dir = "ILSVRC2012/ImageNet/data/ImageNet2012/Val"
    val_datasets = datasets.ImageFolder(val_dir, transform=IMAGENET_TRAIN_TRANSFORM)
    val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=True,num_workers=8,pin_memory=True)

    val(model,val_dataloader)