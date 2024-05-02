import os
import time
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary
from matplotlib import pyplot as plt

from dataloading import data_load
from autoencoder_fully import AutoEncoder
from train_val_test import train, val, test


def main():
    
    df = pd.read_csv('/home/akahige/Python Work/Denoising/tests_fully.csv')
    
    test_ids = list(df['test_id'])
    
    for test_id in test_ids:
        row = int(test_id - 1)
        EPOCHS = int(df['epochs'][row])
        BATCH = int(df['batch_size'][row])
        LR = float(df['learning_rate'][row])
        AMSGRAD = bool(df['amsgrad'][row])
        VAL_SIZE = 5000
        SAVE_PATH = f'/home/akahige/Python Work/Denoising/archive/model_ckpts/fully_cnn_{test_id}/'
    
        print("================== Parameters ====================")
        print(f"Test Number {test_id}")
        print(f"Epochs: {EPOCHS}")
        print(f"Batch Size: {BATCH}")
        print(f"Learning Rate: {LR}")
        print(f"Use AmsGrad: {AMSGRAD}")
        print(f"Save Destination: {SAVE_PATH}")
        print("==================================================")
            
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
        
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, amsgrad=AMSGRAD)
        
        train_loss = []
        validation_loss = []
        train_psnr = []
        validation_psnr = []
        previous_loss = []
        psnr = []
        start_total_time = time.time()
        for epoch in range(EPOCHS):
            print()
            print(f"============ In Epoch {epoch+1}/{EPOCHS} ==================")
            
            start_train_time = time.time()
            
            tr_loss, tr_psnr = train(model, train_loader, optimizer, loss_fn, device)
            
            end_train_time = time.time()
            train_loss.append(float(tr_loss.cpu()))
            train_psnr.append(float(tr_psnr.cpu()))
            
            train_time = round(end_train_time - start_train_time, 2)
            
            start_val_time = time.time()
            
            
            val_loss, val_psnr = val(model, val_loader, loss_fn, device)
            
            end_val_time = time.time()
            validation_loss.append(float(val_loss.cpu()))
            validation_psnr.append(float(val_psnr.cpu()))
            
            val_time = round(end_val_time - start_val_time, 2)
            
            end_epoch_time = time.time()
            epoch_time = round(end_epoch_time - start_train_time, 2)
            
            if epoch == 0:
                previous_loss.append(val_loss)
                psnr.append(val_psnr)
            elif epoch > 0:
                if val_loss < previous_loss[0]:
                    previous_loss.pop()
                    previous_loss.append(val_loss)
                    
                    psnr.pop()
                    psnr.append(val_psnr)
                    
                    torch.save(model.state_dict(), f'{SAVE_PATH}{test_id}.pt')

            print()
            print(f"=========== Total Epoch Runtime {epoch_time} s ====================")
            
        end_total_time = time.time()
        
        total_time = round((end_total_time - start_total_time) / 60, 2)
        
        print()
        print("==========================================================")
        print(f"========= Total Runtime {total_time} minutes !===========")
        print("==========================================================")
        
        print(f"Loss saved: {float(previous_loss[0].cpu())}")
        print(f"PSNR saved: {float(psnr[0].cpu())}")
        
        df.loc[row, 'Val Loss'] = float(previous_loss[0].cpu())
        df.loc[row, 'Val PSNR'] = float(psnr[0].cpu())
        
        df.to_csv('/home/akahige/Python Work/Denoising/tests_fully.csv', index=False)
        
        epochs = [i for i in range(1, EPOCHS + 1)]
        
        fig = plt.figure(figsize=(10, 10))
        plt.plot(epochs, train_loss, label='Train Loss')
        plt.plot(epochs, validation_loss, label='Validation Loss')
        
        plt.xlabel('Number of Epochs')
        plt.ylabel('Loss')
        plt.suptitle('Losses')
        plt.legend()
        
        plt.savefig(f'{SAVE_PATH}{test_id}_loss.png')
        # plt.show()
        
        fig = plt.figure(figsize=(10, 10))
        plt.plot(epochs, train_psnr, label='Train PSNR')
        plt.plot(epochs, validation_psnr, label='Validation PSNR')
        
        plt.xlabel('Number of Epochs')
        plt.ylabel('PSNR (dB)')
        plt.suptitle('PSNR Score')
        plt.legend()
        
        plt.savefig(f'{SAVE_PATH}{test_id}_psnr.png')
        # plt.show()
    
    
if __name__ == "__main__":
    
    main()
    