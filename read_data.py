from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import torch

def read_cifar10(batchsize,data_dir):

    transform_train = transforms.Compose([
                                    # transforms.RandomRotation(),  
                                    transforms.RandomCrop(32, padding=4),  
                                    transforms.RandomHorizontalFlip(p=0.5), 
                                    # transforms.Resize((32,32)),
                                    transforms.ToTensor(),
                                    # transforms.ColorJitter(brightness=1),  
                                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])
                                    ])

    transform_test = transforms.Compose([
                                    # transforms.Resize((32,32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])
                                    ])

    data_train = datasets.CIFAR10(root=data_dir,
                                  train=True,
                                  transform=transform_train, 
                                  #target_transform=transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)),
                                  download=True)

    data_test = datasets.CIFAR10(root=data_dir,
                                 train=False,
                                 transform=transform_test,
                                 #target_transform=transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)),
                                 download=True
                                 )

    data_loader_train = DataLoader(dataset=data_train,
                                   batch_size=batchsize,
                                   shuffle=True,
                                   pin_memory=True
                                   )
    data_loader_test = DataLoader(dataset=data_test,
                                  batch_size=batchsize,
                                  shuffle=False,
                                  pin_memory=True
                                  )
    return data_loader_train,data_loader_test

def read_cifar100(batchsize,data_dir,ratio):

    print('The CIFAR100 dataset is chosen')
    CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)  
    stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))  

    transform_train = transforms.Compose([
                                    # transforms.RandomRotation(),  
                                    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),  
                                    transforms.RandomHorizontalFlip(), 
                                    #transforms.RandomRotation(15),
                                    transforms.ToTensor(), 
                                    #transforms.Normalize(mean=CIFAR100_TRAIN_MEAN,std=CIFAR100_TRAIN_STD)
                                    transforms.Normalize(*stats)
                                    ])

    transform_test = transforms.Compose([
                                    # transforms.Resize((32,32)),
                                    transforms.ToTensor(),
                                    #transforms.Normalize(mean=CIFAR100_TRAIN_MEAN,std=CIFAR100_TRAIN_STD)
                                    transforms.Normalize(*stats)
                                    ])

    data_train = datasets.CIFAR100(root=data_dir,
                                  train=True,
                                  transform=transform_train,
                                  download=True)

    data_test = datasets.CIFAR100(root=data_dir,
                                 train=False,
                                 transform=transform_test,
                                 download=True
                                 )

    data_loader_train = DataLoader(dataset=data_train,
                                   batch_size=batchsize,
                                   shuffle=True,
                                   pin_memory=True
                                   )
    data_loader_test = DataLoader(dataset=data_test,
                                  batch_size=batchsize,
                                  shuffle=False,
                                  pin_memory=True
                                  )
    return data_loader_train,data_loader_test


def read_mnist(batchsize,data_dir):

    transform_train = transforms.Compose([
                                    # transforms.RandomRotation(),  
                                    # transforms.RandomCrop(32, padding=4),  
                                    # transforms.RandomHorizontalFlip(p=0.5), 
                                    # transforms.Resize((32,32)),
                                    transforms.ToTensor(),
                                    # transforms.ColorJitter(brightness=1),  
                                    transforms.Normalize((0.1307,), (0.3081,))
                                    ])

    transform_test = transforms.Compose([
                                    # transforms.Resize((32,32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                    ])

    data_train = datasets.MNIST(root=data_dir,
                                  train=True,
                                  transform=transform_train,
                                  download=True)

    data_test = datasets.MNIST(root=data_dir,
                                 train=False,
                                 transform=transform_test,
                                 download=True
                                 )

    data_loader_train = DataLoader(dataset=data_train,
                                   batch_size=batchsize,
                                   shuffle=True,
                                   pin_memory=True
                                   )
    data_loader_test = DataLoader(dataset=data_test,
                                  batch_size=batchsize,
                                  shuffle=False,
                                  pin_memory=True
                                  )
    return data_loader_train,data_loader_test