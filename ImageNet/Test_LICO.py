import torch
import torch.nn.functional as F
from Mydataset_LICO import get_trainset,get_valset




def test(quantization_model,args):
    test_loader=get_valset(128)
    quantization_model.to(args.device)
    total_correct=0
    total_loss=0
    with torch.no_grad():
        for images,labels in test_loader:
            images=images.to(args.device)
            labels=labels.to(args.device)
            preds=quantization_model(images)
            loss=F.cross_entropy(preds,labels)
            a=preds.argmax(dim=1).equal(labels)
            total_loss+=loss.item()
            total_correct+=preds.argmax(dim=1).eq(labels).sum().item()
        test_acc=total_correct/args.valset_length
    return test_acc
if __name__=='__main__':
   print(test())