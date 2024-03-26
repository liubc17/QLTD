import torch.nn as nn
from torch.nn import functional as F
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from calculate_model_size.model_size import compute_model_nbits
from get_cifar10_pretrained_model_new import ResNet18
from ResNet18_Baidu import ResNet18_Baidu

def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct
def main():

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
    cifar10_test = DataLoader(CIFAR10(root='./CIFAR10',
                                      train=False,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          normalize,
                                      ]), download=True), batch_size=100,
                              shuffle=False)
    # device = torch.device('cuda:0')
    # model = ResNet18_Baidu()
    # state = torch.load('save_model/Baiducloud/model9509.pkl')
    # param = state['network']
    # model.load_state_dict(param)
    # model.to(device)
    # print(model)
    quantization_compressed_model = torch.load('save_model/no_shortcut/parameters/centroids/quantization_1.6_2_k2048_epoch90_lr0.001_gamma0.001_compressed_model_cifar10_trained.pt')
    print(quantization_compressed_model)
    print('the bits of model with low-rank decomposition and quantization:',compute_model_nbits(quantization_compressed_model))
    device = torch.device('cuda:0')
    quantization_compressed_model = quantization_compressed_model.to(device)
    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(quantization_compressed_model.parameters(), lr=1e-3)
    for epoch in range(1):
        test_loss = 0
        test_acc = 0
        num_correct = 0
        for batchidx, (x, label) in enumerate(cifar10_test):
            x, label = x.to(device), label.to(device)
            y_ = quantization_compressed_model(x)
            loss = criteon(y_, label)
            test_loss += loss.item()
            num_correct += get_acc(y_, label)
        print("epoch:%d,test_loss:%f,test_acc:%f" % (epoch, test_loss / len(cifar10_test),
                                                       num_correct / 10000))  # 打印的是最后一个batch的loss
        print(num_correct)
    #
    # for bachidx, (x, label) in enumerate(cifar10_test):
    #     x, label = x.to(device), label.to(device)
    #     y_ = quantization_compressed_model(x)
    #     test_acc = get_acc(y_, label)
    #     print('acc',test_acc)
if __name__ == "__main__":
    main()