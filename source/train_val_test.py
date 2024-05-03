import torch
from torchviz import make_dot
from matplotlib import pyplot as plt

from noisy import add_noise
from noise_metrics import PSNR
from torcheval.metrics.functional import multiclass_f1_score
from transformers import ViTFeatureExtractor, ViTForImageClassification


LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'

def train(net, train_loader, optimizer, loss_fn, device):
    running_loss = torch.tensor([0.0]).to(device)
    running_psnr = torch.tensor([0.0]).to(device)
    net.train()
    step = 0
    for img, labels in train_loader:
        img = img.to(device)
        image_noisy = add_noise(img).to(device)
        
        optimizer.zero_grad()
        output = net(image_noisy).to(device)
        
        running_psnr += PSNR(img, output)
        
        loss = loss_fn(output, img).to(device)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        step += 1
        
        if step < len(train_loader):
            print(f"Train Step {step}/{len(train_loader)}")
            print(LINE_UP, end=LINE_CLEAR)
    
    train_loss = running_loss / len(train_loader)
    
    psnr_score = running_psnr / len(train_loader)
    
    print(f"Train Step {step}/{len(train_loader)} --- Train Loss: {float(train_loss.cpu())} ---- Train PSNR: {float(psnr_score.cpu())}")
    
    return train_loss, psnr_score


def val(net, val_loader, loss_fn, device):
    net.eval()
    with torch.no_grad(): 
        running_loss = torch.tensor([0.0]).to(device) 
        running_psnr = torch.tensor([0.0]).to(device)
        step = 0
        for img, labels in val_loader:
            img = img.to(device)
            image_noisy = add_noise(img).to(device)
            
            output = net(image_noisy).to(device)
            
            running_psnr += PSNR(img, output)
            
            loss = loss_fn(output, img).to(device)
            running_loss += loss.item()
            step += 1
            
            if step < len(val_loader):
                print(f" Val Step {step}/{len(val_loader)}")
                print(LINE_UP, end=LINE_CLEAR)
            
        val_loss = running_loss / len(val_loader)
        
        psnr_score = running_psnr / len(val_loader)
        
    print(f"Val Step {step}/{len(val_loader)} ------- Val Loss: {float(val_loss.cpu())} ---- Val PSNR: {float(psnr_score.cpu())}")
    
    return val_loss, psnr_score


def test(net, test_loader, device):
    
    # raise ValueError("Forgot to change the MEAN and STD values based on the normalization chosen!")
    # net.eval()
    # MEAN = torch.tensor([0.5, 0.5, 0.5]).to(device)
    # STD = torch.tensor([0.5, 0.5, 0.5]).to(device)
    
    # feature_extractor = ViTFeatureExtractor.from_pretrained('nateraw/vit-base-patch16-224-cifar10')
    # model = ViTForImageClassification.from_pretrained('nateraw/vit-base-patch16-224-cifar10').to(device)
    # f1 = torch.tensor([0.0]).to(device)
    with torch.no_grad():
        for img, labels in test_loader:
            img = img.to(device)
            labels = labels.to(device)
            noisy = add_noise(img)
            output = net(noisy).to(device)
        
            images = img[:, :, :, :].permute((0, 2, 3, 1)).cpu()
            noisy_images = noisy[:, :, :, :].permute((0, 2, 3, 1)).cpu()
            output_images = output[:, :, :, :].permute((0, 2, 3, 1)).cpu()
            
            for i in range(32):
                fig, axs = plt.subplots(1, 3)
                axs[0].imshow(images[i, :, :, :])
                axs[1].imshow(noisy_images[i, :, :, :])
                axs[2].imshow(output_images[i, :, :, :])
                plt.show()
            
            exit()
            
    #         inputs = feature_extractor(images=output, return_tensors="pt", do_rescale=False).to(device)
    #         preds = model(**inputs).logits.argmax(dim=1).to(device)
            
    #         f1 += multiclass_f1_score(preds, labels, num_classes=10, average='weighted').to(device)
            
    #         print(f"Predictions: {preds.cpu()}")
    #         print(f"Truth: {labels.cpu()}")
                
        
    # total_f1 = f1 / len(test_loader)
            # print(img[0, :, :, :].permute(1, 2, 0).shape)
            # exit()
    # print(f"Final f1: {total_f1}")
            
            # img = img[33] * STD[:, None, None] + MEAN[:, None, None]
            # img = noisy[33]
            # img = img.permute(1, 2, 0).cpu()
            # for i in range(img.shape[0]):
            #     fig, axs = plt.subplots(1, 3, figsize=(10, 10))
            #     axs[0].imshow(img[i, :, :, :].permute(1, 2, 0).cpu())
            #     axs[1].imshow(noisy[i, :, :, :].permute(1, 2, 0).cpu())
            #     axs[2].imshow(output[i, :, :, :].permute(1, 2, 0).cpu())
            #     plt.show()
            # exit()
        
        
    