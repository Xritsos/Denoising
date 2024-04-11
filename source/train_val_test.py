import torch
from matplotlib import pyplot as plt
import torchvision.transforms as transforms

LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'

def train(net, train_loader, optimizer, loss_fn, device):
    running_loss = 0.0
    net.train()
    step = 0
    for img, _ in train_loader:
        img = img.to(device)
        optimizer.zero_grad()
        output = net(img)
        loss = loss_fn(output, img)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        step += 1
        
        print(f"Step {step}/{len(train_loader)}")
        
        if step < len(train_loader):
            print(LINE_UP, end=LINE_CLEAR)
    
    train_loss = running_loss / len(train_loader)

    return train_loss


def val(net, val_loader, loss_fn, device):
    net.eval()
    with torch.no_grad(): 
        running_loss = 0.0 
        step = 0
        for img, _ in val_loader:
            img = img.to(device)
            # image_noisy = add_noise(image_batch,noise_factor)
            # image_noisy = image_noisy.to(device)
            output = net(img)
            loss = loss_fn(output, img)
            running_loss += loss.item()
            step += 1
        
            print(f"Step {step}/{len(val_loader)}")
            
            if step < len(val_loader):
                print(LINE_UP, end=LINE_CLEAR)
            
        val_loss = running_loss / len(val_loader)
    
    return val_loss


def test(net, test_loader, device):
    net.eval()
    MEAN = torch.tensor([0.5, 0.5, 0.5]).to(device)
    STD = torch.tensor([0.5, 0.5, 0.5]).to(device)
    
    with torch.no_grad():
        for img, _ in test_loader:
            img = img.to(device)
            output = net(img)
            
            img = img[33] * STD[:, None, None] + MEAN[:, None, None]
            img = img.permute(1, 2, 0).cpu()
            
            fig, axs = plt.subplots(1, 2, figsize=(10, 10))
            axs[0].imshow(img)
            axs[1].imshow(output[33].permute(1, 2, 0).cpu())
            
            plt.show()
            exit()
        