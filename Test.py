import os
import time
import torch
import torch.nn.functional as F 
from ResNet18_Baidu import ResNet18
from Mydataset import get_test_data_loader
from torch.utils.tensorboard import SummaryWriter
from calculate_model_size.model_size import compute_model_nbits
MODEL_PATH='Models/model9509.pkl'
TESTSET_LENGTH=10000
device='cuda' if torch.cuda.is_available() else 'cpu'

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def test():
    test_loader=get_test_data_loader(100)
    quantization_model = torch.load('save_fine/QLTD/4_4_0.01_R18_C10.pt')
    quantization_model.to(device)
    # print(quantization_model)
    print('model bits:',compute_model_nbits(quantization_model))
    print('compression ratio:',compute_model_nbits(torch.load('save_model/origin_model/Resnet18_cifar10.pt')) / compute_model_nbits(quantization_model))
    # torch.save_decom(network,'E:/！！！research/Holiday_网络压缩/LRQ/save_model/origin_model/Resnet18_cifar10.pt')
    total_correct=0
    total_loss=0
    with torch.no_grad():
        start = time.time()
        for images,labels in test_loader:
            images=images.to(device)
            labels=labels.to(device)
            preds=quantization_model(images)
            loss=F.cross_entropy(preds,labels)
            a=preds.argmax(dim=1).equal(labels)
            total_loss+=loss.item()
            total_correct+=preds.argmax(dim=1).eq(labels).sum().item()
        time_used = time.time() - start
        print(time_used)
        test_acc=total_correct/TESTSET_LENGTH
    print(test_acc*100)
    return test_acc
if __name__=='__main__':
    test()

