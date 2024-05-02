import torch
import pandas as pd

from train_val_test import test
from dataloading import data_load
from autoencoder_fully import AutoEncoder

from matplotlib import pyplot as plt
from transformers import ViTFeatureExtractor, ViTForImageClassification
from torch import nn
from torcheval.metrics.functional import multiclass_f1_score


classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
           'horse', 'ship', 'truck']


def test_classifiers():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader, test_loader, _ = data_load(validation_size=100,
                                                         batch_size=4, 
                                                         visualize_split=False)
    
   
    feature_extractor = ViTFeatureExtractor.from_pretrained('nateraw/vit-base-patch16-224-cifar10')
    model = ViTForImageClassification.from_pretrained('nateraw/vit-base-patch16-224-cifar10').to(device)
    f1 = torch.tensor([0.0]).to(device)
    for imgs, labels in val_loader:
        imgs = imgs.to(device) 
        labels = labels.to(device)
        
        inputs = feature_extractor(images=total_set, return_tensors="pt", do_rescale=False).to(device)
        preds = model(**inputs).logits.argmax(dim=1).to(device)
        
        f1 += multiclass_f1_score(preds, total_labels, num_classes=10, average='weighted').to(device)
            
        
    total_f1 = f1 / len(val_loader)
    
    print(f"Final f1: {total_f1}")
    

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
    
    # test_model()
    print()
    test_classifiers()
    