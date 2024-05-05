import torch

from noisy import add_noise
from noise_metrics import PSNR


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
            