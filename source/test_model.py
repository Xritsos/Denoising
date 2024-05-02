import torch
import pandas as pd

from train_val_test import test
from dataloading import data_load
from autoencoder_fully import AutoEncoder


def test_model():
    test_id = 3
    row = test_id - 1
    df = pd.read_csv('/home/akahige/Python Work/Denoising/tests_fully.csv')
    
    EPOCHS = int(df['epochs'][row])
    BATCH = int(df['batch_size'][row])
    LR = float(df['learning_rate'][row])
    AMSGRAD = bool(df['amsgrad'][row])
    VAL_SIZE = 5000
    
    print(f"Val Loss: {df['Val Loss'][row]}")
    print(f"Val PSNR: {df['Val PSNR'][row]}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader, test_loader, _ = data_load(validation_size=VAL_SIZE,
                                                         batch_size=BATCH, 
                                                         visualize_split=False)
    
    # load best model for testing
    model = AutoEncoder().to(device)
    model.load_state_dict(torch.load(f'/home/akahige/Python Work/Denoising/archive/model_ckpts/fully_cnn_{test_id}/{test_id}.pt'))
   
    test(model, test_loader, device)
    
    
if __name__ == "__main__":
    
    test_model()
    