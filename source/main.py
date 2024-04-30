import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary
from matplotlib import pyplot as plt

from dataloading import data_load
from autoencoder_fully import AutoEncoder
from train_val_test import train, val, test


def main():
    test_id = 1 # it will be read from file automatically
    EPOCHS = 30
    BATCH = 256
    LR = 1e-3
    VAL_SIZE = 5000
    SAVE_PATH = f'/home/akahige/Python Work/Denoising/archive/model_ckpts/fully_cnn_{test_id}/'
    
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
     
    torch.manual_seed(7)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader, test_loader, _ = data_load(validation_size=VAL_SIZE,
                                                         batch_size=BATCH, 
                                                         visualize_split=False)
    print()
    print(f"Available Device: {device}")
    print("Proceeding with training...")
    
    model = AutoEncoder().to(device)
    
    # summary(model, (3, 32, 32))
    # exit()
    
    loss_fn = nn.MSELoss()
    loss_fn.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    train_loss = []
    validation_loss = []
    previous_loss = []
    start_total_time = time.time()
    for epoch in range(EPOCHS):
        print()
        print(f"============ In Epoch {epoch+1}/{EPOCHS} ==================")
        
        start_train_time = time.time()
        
        tr_loss = train(model, train_loader, optimizer, loss_fn, device)
        
        end_train_time = time.time()
        train_loss.append(float(tr_loss.cpu()))
        
        train_time = round(end_train_time - start_train_time, 2)
        
        start_val_time = time.time()
        
        
        val_loss = val(model, val_loader, loss_fn, device)
        
        end_val_time = time.time()
        validation_loss.append(float(val_loss.cpu()))
        
        val_time = round(end_val_time - start_val_time, 2)
        
        end_epoch_time = time.time()
        epoch_time = round(end_epoch_time - start_train_time, 2)
        
        if epoch == 0:
            previous_loss.append(val_loss)
        elif epoch > 0:
            if val_loss < previous_loss[0]:
                previous_loss.pop()
                previous_loss.append(val_loss)
                
                torch.save(model.state_dict(), f'{SAVE_PATH}{test_id}.pt')

        print()
        print(f"=========== Total Epoch Runtime {epoch_time} s ====================")
        
    end_total_time = time.time()
    
    total_time = round((end_total_time - start_total_time) / 60, 2)
    
    print()
    print("==========================================================")
    print(f"========= Total Runtime {total_time} minutes !===========")
    print("==========================================================")
    
    print(f"Loss saved: {previous_loss[0]}")
    
    epochs = [i for i in range(1, EPOCHS + 1)]
    
    fig = plt.figure(figsize=(10, 10))
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, validation_loss, label='Validation Loss')
    
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')
    plt.suptitle('Losses')
    plt.legend()
    
    plt.savefig(f'{SAVE_PATH}{test_id}_loss.png')
    plt.show()
        
        
    # load best model for testing
    model = torch.load(f'{SAVE_PATH}{test_id}.pt')
    test(model, test_loader, device)
    
    
if __name__ == "__main__":
    
    main()
    