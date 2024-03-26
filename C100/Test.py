import torch
import torch.nn.functional as F 
from Mydataset import get_test_data_loader
from torch.utils.tensorboard import SummaryWriter
from resnetforcifar import resnet18cifar
from calculate_model_size.model_size import compute_model_nbits
MODEL_PATH='Models/model9509.pkl'
TESTSET_LENGTH=10000
device='cuda:1' if torch.cuda.is_available() else 'cpu'


def test():
    test_loader=get_test_data_loader(128)
    ckpt = 'save_model/origin_model/Resnet18_cifar100.pt'
    original_model = torch.load(ckpt)
    quantization_model = torch.load('save_fine/QLTD/4_2_0.005_512_R18_C100.pt')
    # print(quantization_model)
    quantization_model.to(device)
    print('compression ratio:', compute_model_nbits(original_model) / compute_model_nbits(quantization_model))
    # torch.save_decom(network,'E:/！！！research/Holiday_网络压缩/LRQ/save_model/origin_model/Resnet18_cifar10.pt')
    total_correct=0
    total_loss=0
    with torch.no_grad():
        for images,labels in test_loader:
            images=images.to(device)
            labels=labels.to(device)
            preds=quantization_model(images)
            loss=F.cross_entropy(preds,labels)
            a=preds.argmax(dim=1).equal(labels)
            total_loss+=loss.item()
            total_correct+=preds.argmax(dim=1).eq(labels).sum().item()
        test_acc=total_correct/TESTSET_LENGTH
    print(test_acc*100)
    print(ckpt)
    return test_acc


if __name__=='__main__':
   test()



