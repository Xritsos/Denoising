import torch
import pandas as pd
from torch import nn
from torcheval.metrics.functional import multiclass_f1_score
from transformers import ViTFeatureExtractor, ViTForImageClassification

from train_val_test import test
from dataloading import data_load
from model_c import Model_C
from model_b import Model_B
from noisy import add_noise


classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
           'horse', 'ship', 'truck']


def test_classifiers():
    test_id = 37
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader, test_loader, _ = data_load(validation_size=100,
                                                         batch_size=4, 
                                                         visualize_split=False)
    
    # load best model for testing
    model = Model_C().to(device)
    model.load_state_dict(torch.load(f'./archive/model_ckpts/fully_cnn_{test_id}/{test_id}.pt'))
    model.eval()
   
    feature_extractor = ViTFeatureExtractor.from_pretrained('nateraw/vit-base-patch16-224-cifar10')
    classifier = ViTForImageClassification.from_pretrained('nateraw/vit-base-patch16-224-cifar10').to(device)
    f1 = torch.tensor([0.0]).to(device)
    
    for imgs, labels in test_loader:
        imgs = imgs.to(device) 
        labels = labels.to(device)
        
        noisy = add_noise(imgs).to(device)
        
        output = model(noisy)
        
        inputs = feature_extractor(images=output, return_tensors="pt", do_rescale=False).to(device)
        preds = classifier(**inputs).logits.argmax(dim=1).to(device)
        
        f1 += multiclass_f1_score(preds, labels, num_classes=10, average='macro').to(device)
        
        
    total_f1 = f1 / len(test_loader)
    
    print(f"Final f1: {total_f1}")
    

def test_model():
    test_id = 36
    row = test_id - 1
    df = pd.read_csv('./tests_fully.csv')
    
    EPOCHS = int(df['epochs'][row])
    BATCH = int(df['batch_size'][row])
    LR = float(df['learning_rate'][row])
    AMSGRAD = bool(df['amsgrad'][row])
    VAL_SIZE = 5000
    BATCH = 32
    print(f"Val Loss: {df['Val Loss'][row]}")
    print(f"Val PSNR: {df['Val PSNR'][row]}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader, test_loader, _ = data_load(validation_size=VAL_SIZE,
                                                         batch_size=BATCH, 
                                                         visualize_split=False)
    
    # load best model for testing
    model = AutoEncoder().to(device)
    model.load_state_dict(torch.load(f'./archive/model_ckpts/fully_cnn_{test_id}/{test_id}.pt'))
   
    test(model, test_loader, device)
    
    
if __name__ == "__main__":
    
    # test_model()
    # print()
    test_classifiers()
    