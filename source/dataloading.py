import torch
import torchvision
import numpy as np
from torch.utils.data import random_split
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader

from plotting import visualize_sets


    
def data_load(validation_size=5000, training_size=20000, batch_size=256, visualize_split=False):
    
    torch.manual_seed(7)
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    data_set = torchvision.datasets.CIFAR10(root='./data', 
                                            train=True,
                                            download=True, 
                                            transform=transform)
    
    test_set = torchvision.datasets.CIFAR10(root='./data',
                                            train=False,
                                            transform=transform)
    
    if len(data_set) > validation_size:
        train_size = len(data_set) - validation_size
    else:
        raise ValueError("Validation set should be smaller than training set !")
    
    # split to train and validation sets
    train_set, val_set = random_split(data_set, [train_size, validation_size])
    
    
    # keep only a trainig size of your preference
    idxs = np.random.randint(0, len(train_set), training_size)
    
    train_set = torch.utils.data.Subset(train_set, idxs)
    
    print()
    print(f"Train Set Size: {len(train_set)}")
    print(f"Validation Set Size: {len(val_set)}")
    print(f"Test Set Size: {len(test_set)}")
    
    classes = data_set.classes
        
    if visualize_split == True:
        visualize_sets(train_set, val_set, test_set, classes)
    
    train_loader = DataLoader(train_set, batch_size, 
                              shuffle=True, 
                              num_workers=8,
                              prefetch_factor=2,
                              persistent_workers=True, 
                              pin_memory=True)
    
    val_loader = DataLoader(val_set, batch_size,
                            shuffle=True, 
                            num_workers=8,
                            prefetch_factor=2,
                            persistent_workers=True, 
                            pin_memory=True)
    
    test_loader = DataLoader(test_set, batch_size, 
                             num_workers=8,
                             shuffle=True,
                             prefetch_factor=2,
                             persistent_workers=True, 
                             pin_memory=True)
    
    return train_loader, val_loader, test_loader, classes
    