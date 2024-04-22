import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt

from dataloading import data_load
from autoencoder import AutoEncoder
from train_val_test import train, val, test


def main():
    EPOCHS = 10
    BATCH = 256
    LR = 1e-3
    VAL_SIZE = 5000
    
    torch.manual_seed(7)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader, test_loader, _ = data_load(validation_size=VAL_SIZE,
                                                         batch_size=BATCH, 
                                                         visualize_split=False)
    print()
    print(f"Available Device: {device}")
    print("Proceeding with training...")
    
    model = AutoEncoder().to(device)
    
    loss_fn = nn.MSELoss()
    loss_fn.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    train_loss = []
    validation_loss = []
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
        print()
        print(f"=========== Total Epoch Runtime {epoch_time} s ====================")
        
    end_total_time = time.time()
    
    total_time = round((end_total_time - start_total_time) / 60, 2)
    
    print()
    print("==========================================================")
    print(f"========= Total Runtime {total_time} minutes !===========")
    print("==========================================================")
    
    epochs = [i for i in range(1, EPOCHS + 1)]
    
    fig = plt.figure(figsize=(10, 10))
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, validation_loss, label='Validation Loss')
    
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')
    plt.suptitle('Losses')
    plt.legend()
    
    plt.show()
        
    test(model, test_loader, device)
    
    
if __name__ == "__main__":
    
    main()
    