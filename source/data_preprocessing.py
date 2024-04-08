import sys
import time

import torch
import torchvision
import numpy as np
from torch.utils.data import random_split
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from matplotlib import pyplot as plt

sys.path.append('./')
from source.plotting import visualize_sets
from source.autoencoder import Encoder, Decoder


def add_noise(inputs,noise_factor=0.3):
	noisy = inputs+torch.randn_like(inputs) * noise_factor
	noisy = torch.clip(noisy,0.,1.)
	return noisy
    
    
def data_load(validation_size=10000, visualize_split=False):
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
    
    print()
    print(f"Train Set Size: {len(train_set)}")
    print(f"Validation Set Size: {len(val_set)}")
    print(f"Test Set Size: {len(test_set)}")
    
    classes = data_set.classes
        
    if visualize_split == True:
        visualize_sets(train_set, val_set, test_set, classes)
    
    
    return train_set, val_set, test_set, classes
    
    
def train(train_set, val_set, test_set, classes):
    batch_size = 2048
    
    train_loader = DataLoader(train_set, batch_size, 
                              shuffle=True, 
                              num_workers=7, 
                              pin_memory=True)
    
    val_loader = DataLoader(val_set, batch_size*2, 
                            num_workers=7, 
                            pin_memory=True)
    
    test_loader = DataLoader(test_set, batch_size*2, 
                             num_workers=7, 
                             pin_memory=True)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    print()
    print(f"======= Device Used: {device} =============")
        
    loss_fn = torch.nn.MSELoss()
    lr = 1e-4
    d = 4
    noise_factor = 0.2
    
    torch.manual_seed(7)
    
    encoder = Encoder(num_input_channels=3, base_channel_size=8, latent_dim=128).to(device)
    decoder = Decoder(num_input_channels=3, base_channel_size=8, latent_dim=128).to(device)
    
    # print(decoder)
    # exit()
    
    params_to_optimize = [
	    {'params': encoder.parameters()},
	    {'params': decoder.parameters()}
    ]

    optimizer = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-04)
    
    train_len = len(train_loader)
    val_len = len(val_loader)
    
    start_training = time.time()
    
    val_loss = []
    train_loss = []
    
    for epoch in range(60):

        encoder.to(device)
        decoder.to(device)
        encoder.train()
        decoder.train()
    
        print()
        print(f"============ In epoch {epoch}/20.... =================")
        
        start_train_epoch = time.time()
        ind = 0
        conc_out = []
        conc_label = []
        # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
        for image_batch, _ in train_loader: # with "_" we just ignore the labels (the second element of the dataloader tuple)
            # Move tensor to the proper device
            print()
            print(f"In step {ind}/{train_len}")
            image_noisy = add_noise(image_batch, noise_factor)
            image_batch = image_batch.to(device)
            image_noisy = image_noisy.to(device)
            # Encode data
            encoded_data = encoder(image_noisy)
            # Decode data
            decoded_data = decoder(encoded_data)
            # Evaluate loss
            loss = loss_fn(decoded_data, image_batch)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())
            
            ind += 1
                
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label)
        # Evaluate global loss
        train_loss.append(loss_fn(conc_out, conc_label).detach().cpu().numpy())
            
        end_train_epoch = time.time()
        
        print()
        print(f"Epoch Training Time: {end_train_epoch - start_train_epoch}")
            
        start_val_epoch = time.time()
        
        encoder.eval()
        decoder.eval()
        with torch.no_grad(): # No need to track the gradients
            # Define the lists to store the outputs for each batch
            conc_out = []
            conc_label = []
            ind_2 = 0
            for image_batch, _ in val_loader:
                print()
                print(f"In step {ind_2}/{val_len}")
                # Move tensor to the proper device
                image_noisy = add_noise(image_batch,noise_factor)
                image_noisy = image_noisy.to(device)
                # Encode data
                encoded_data = encoder(image_noisy)
                # Decode data
                decoded_data = decoder(encoded_data)
                # Append the network output and the original image to the lists
                conc_out.append(decoded_data.cpu())
                conc_label.append(image_batch.cpu())
                ind_2 += 1
                # val_loss_all.append(loss_fn(decoded_data, image_batch).detach().cpu().numpy())
            # Create a single tensor with all the values in the lists
            conc_out = torch.cat(conc_out)
            conc_label = torch.cat(conc_label)
            # Evaluate global loss
            val_loss.append(loss_fn(conc_out, conc_label))
        
        end_val_epoch = time.time()
        
        print()
        print(f"Epoch Validation Time: {end_val_epoch - start_val_epoch}")
  
    end_training = time.time()
    print()
    print(f"Total Runtime for 20 Epochs: {end_training - start_training}")
    
    encoder.eval()
    decoder.eval()
    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        for image_batch, _ in test_loader:
            print()
            print(f"In step {ind_2}/{val_len}")
            # Move tensor to the proper device
            image_noisy = add_noise(image_batch,noise_factor)
            image_noisy = image_noisy.to(device)
            # Encode data
            encoded_data = encoder(image_noisy)
            # Decode data
            decoded_data = decoder(encoded_data)
            
            fig, axs = plt.subplots(1, 3)
            
            axs[0].imshow(image_batch[0].permute((1, 2, 0)).detach().cpu().numpy())
            axs[1].imshow(image_noisy[0].permute((1, 2, 0)).detach().cpu().numpy())
            axs[2].imshow(decoded_data[0].permute((1, 2, 0)).detach().cpu().numpy())
            plt.suptitle(f"Label {classes[_[0]]}")
            
            plt.show()
            break
    
    return train_loss, val_loss
  
    
if __name__ == "__main__":
    
    train_set, val_set, test_set, classes = data_load(7000)
    
    train_loss, val_loss = train(train_set, val_set, test_set, classes)
    
    fig = plt.figure()
    
    plt.plot(train_loss)
    plt.plot(val_loss)
    
    plt.show()