import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import warnings

import torch
import torch.nn as nn
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from read_data import read_cifar100
from models import *
from insgd import INSGD
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch Cifar100 Training')

parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-rt', '--ratio', default=1, type=int, dest='ratio')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-m', '--model', default='wresnet', type=str,
                    dest='model', help='Which model to use. The option to choose are '
                                        'resnet18(default), resnet50, mobilenetv3, densenet, pyramidnet'
                                        'cait, convmixer, mlpmixer, vgg, vit, vit_small, swin')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--gamma', default=0.2, type=float,
                    metavar='G', dest='gamma')
parser.add_argument('--patients', default=20, type=float,
                    metavar='G', dest='patients')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--save_dir', type=str, default='./saved')
parser.add_argument('--nsgd', type=int, default=1, dest='nsgd',
                    help='Which optimizer to use. 1 for NISGD (default), 2 for Adam, anything else for SGD')
parser.add_argument('--norm_t', type=int, default=2, dest='norm_t',
                    help='L1-L2 Norm')


def main():
    
    args = parser.parse_args()
    
    seed = args.seed
    if seed is None:
        seed = 0
    
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
            
    main_worker(args)


def main_worker(args):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    data_dir = '../data'
    ratio = args.ratio
    batchsize = args.batch_size
    n_epochs = args.epochs
    model_type = args.model

    # CIFAR-10 dataloader generator
    train_generator,test_generator = read_cifar100(batchsize,data_dir,ratio)

    # Model type
    if model_type == 'resnet18':
        from models.resnet import resnet18
        model = resnet18().to(device)
    elif model_type == 'resnet50':
        from models.resnet import resnet50
        model = resnet50().to(device)
    elif model_type == 'resnet_flat':
        from models.resnet_flat import ResNet20_Flat
        model = ResNet20_Flat().to(device)
    elif model_type == 'wresnet':
        from models.resnet_flat import Wide_ResNet18_Flat
        model = Wide_ResNet18_Flat().to(device)
    elif model_type == 'resnet34':
        from models.resnet import resnet34
        model = resnet34(num_classes=100).to(device)
    elif model_type == 'pyramidnet':
        from models.pyramidnet import PyramidNet
        model = PyramidNet('cifar100', 110, 48, 100).to(device)
    elif model_type == 'simpnet':
        from models.simpnet import simpnet
        model = simpnet(classes=100).to(device)
    elif model_type == 'mlpmixer':
        from models.mlpmixer import MLPMixer
        model = MLPMixer(image_size=32, channels=12, patch_size=2, dim=256, depth=5, num_classes=10).to(device)
    '''
    TO DO (Rest of the Models)
    '''

    #print(model)
    print(device)
    print("There are", sum(p.numel() for p in model.parameters()), "parameters.")
    print("There are", sum(p.numel() for p in model.parameters() if p.requires_grad), "trainable parameters.")
    
    inputs = {}
    
    def get_inputs(name):
        def hook(model, input, output):
            inputs[name] = input[0].detach()
        return hook
      
    for name,module in model.named_modules():
        for param_name, param in module.named_parameters():
            param.input_to_norm = None
        if name != '' and isinstance(module, (nn.Linear, nn.Conv2d)):
            module.register_forward_hook(get_inputs(name))
    

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().to(device)
    
    if args.nsgd == 1:
        print('Optimizer is INSGD')
        optimizer = INSGD(model.parameters(), args.lr,
                                    momentum=args.momentum, norm_t=args.norm_t,
                                    weight_decay=args.weight_decay)
    elif args.nsgd == 2:
        print('Optimizer is Adam')
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                     weight_decay=args.weight_decay)
    else:
        print('Optimizer is Traditional SGD')
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                     momentum=args.momentum,
                                     weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[79,119,159],gamma=args.gamma,last_epoch=-1)
    
    idx_best_loss = 0
    idx_best_acc = 0
    
    log_train_loss = []
    log_train_acc = []
    log_test_loss = []
    log_test_acc = []
    log_test_acc5 = []

    for epoch in range(1, n_epochs+1):
        print("===> Epoch {}/{}, learning rate: {}".format(epoch, n_epochs, scheduler.get_last_lr()))
        train_loss, train_acc = train_loop(train_generator, model, criterion, optimizer, device, inputs)
        test_loss, test_acc1, test_acc5 = test_loop(test_generator, model, criterion, device)
        print("Training loss: {:f}, acc: {:f}".format(train_loss, train_acc))
        print("Test loss: {:f}, acc@1: {:f}, acc@5: {:f}".format(test_loss, test_acc1, test_acc5))
        scheduler.step()
        
        log_train_loss.append(train_loss)
        log_train_acc.append(train_acc)
        log_test_loss.append(test_loss)
        log_test_acc.append(test_acc1)
        log_test_acc5.append(test_acc5)
        
        if test_loss <= log_test_loss[idx_best_loss]:
            print("Save loss-best model.")
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'loss_model.pth'))
            idx_best_loss = epoch - 1
        
        if test_acc1 >= log_test_acc[idx_best_acc]:
            print("Save acc-best model.")
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'acc_model.pth'))
            if test_acc1 > log_test_acc[idx_best_acc]:
                idx_best_acc = epoch - 1    
        print("")
        
    print("=============================================================")

    print("Loss-best model training loss: {:f}, acc: {:f}".format(log_train_loss[idx_best_loss], log_train_acc[idx_best_loss]))   
    print("Loss-best model test loss: {:f}, acc@1: {:f}, acc@5: {:f}".format(log_test_loss[idx_best_loss], log_test_acc[idx_best_loss], log_test_acc5[idx_best_loss]))                
    print("Acc-best model training loss: {:4f}, acc: {:f}".format(log_train_loss[idx_best_acc], log_train_acc[idx_best_acc]))  
    print("Acc-best model test loss: {:f}, acc@1: {:f}, acc@5: {:f}".format(log_test_loss[idx_best_acc], log_test_acc[idx_best_acc], log_test_acc5[idx_best_acc]))              
    print("Final model training loss: {:f}, acc: {:f}".format(log_train_loss[-1], log_train_acc[-1]))                 
    print("Final model test loss: {:f}, acc@1: {:f}, acc@5: {:f}".format(log_test_loss[-1], log_test_acc[-1], log_test_acc5[-1]))                
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'final_model.pth'))
    
    log_train_loss = np.array(log_train_loss)
    log_train_acc = np.array(log_train_acc)
    log_test_loss = np.array(log_test_loss)
    log_test_acc = np.array(log_test_acc)
    log_test_acc5 = np.array(log_test_acc5)
    np.save(os.path.join(args.save_dir, 'log_train_loss.npy'), log_train_loss)
    np.save(os.path.join(args.save_dir, 'log_train_acc.npy'), log_train_acc)
    np.save(os.path.join(args.save_dir, 'log_test_loss.npy'), log_test_loss)
    np.save(os.path.join(args.save_dir, 'log_test_acc.npy'), log_test_acc)
    np.save(os.path.join(args.save_dir, 'log_test_acc5.npy'), log_test_acc5)
    
def train_loop(dataloader, model, loss_fn, optimizer, device, inputs):
    model.train()
    size = len(dataloader.dataset)
    pbar = tqdm(dataloader, total=int(len(dataloader)))
    count = 0
    train_loss = 0.0
    train_acc = 0.0
    for batch, sample in enumerate(pbar):
        x,labels = sample
        x,labels = x.to(device), labels.to(device)

        outputs = model(x)
        # Input NORM
        for name,module in model.named_modules():
            if name != '' and isinstance(module, (nn.Linear, nn.Conv2d)):
                if isinstance(module, nn.Conv2d):
                    if module.groups != 1:
                        for param_name, param in module.named_parameters():
                            if 'bias' not in param_name:
                                param.input_to_norm = [inputs[name], True]
                    else:
                        for param_name, param in module.named_parameters():
                            if 'bias' not in param_name:
                                param.input_to_norm = [inputs[name], False]
                else:
                    for param_name, param in module.named_parameters():
                        if 'bias' not in param_name:
                            param.input_to_norm = inputs[name]
        loss = loss_fn(outputs, labels)
        _,pred = torch.max(outputs,1)
        num_correct = (pred == labels).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss = loss.item()
        acc = num_correct.item()/len(labels)
        count += len(labels)
        train_loss += loss*len(labels)
        train_acc += num_correct.item()
        pbar.set_description(f"loss: {loss:>f}, acc: {acc:>f}, [{count:>d}/{size:>d}]")
        
    return train_loss/count, train_acc/count
        
def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    size = len(dataloader.dataset)

    pbar = tqdm(dataloader, total=int(len(dataloader)))
    count = 0
    test_loss = 0.0
    test_acc1, test_acc5 = 0.0, 0.0
    with torch.no_grad():
        for batch, sample in enumerate(pbar):
            x,labels = sample
            x,labels = x.to(device), labels.to(device)

            outputs = model(x)
            _, pred = outputs.topk(5, 1, largest=True, sorted=True)

            label = labels.view(labels.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            #compute top 5
            correct_5 = correct[:, :5].sum()

            #compute top1
            correct_1 = correct[:, :1].sum()
            loss = loss_fn(outputs, labels)
            #_,pred = torch.max(outputs,1)
            #num_correct = (pred == labels).sum()
            
            loss = loss.item()
            acc1 = correct_1.item()/len(labels)
            acc5 = correct_5.item()/len(labels)
            count += len(labels)
            test_loss += loss*len(labels)
            test_acc1 += correct_1.item()
            test_acc5 += correct_5.item()
            pbar.set_description(f"loss: {loss:>f}, acc@1: {acc1:>f}, acc@5: {acc5:>f}, [{count:>d}/{size:>d}]")
    
    return test_loss/count, test_acc1/count, test_acc5/count


if __name__ == '__main__':
    main()
