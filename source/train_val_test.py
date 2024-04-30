import torch
from torchviz import make_dot
from matplotlib import pyplot as plt

from noisy import add_noise

LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'

def train(net, train_loader, optimizer, loss_fn, device):
    running_loss = torch.tensor([0.0]).to(device)
    net.train()
    step = 0
    for img, _ in train_loader:
        img = img.to(device)
        image_noisy = add_noise(img)
        image_noisy = image_noisy.to(device)
        optimizer.zero_grad()
        output = net(image_noisy).to(device)
        loss = loss_fn(output, img).to(device)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        step += 1
        
        if step < len(train_loader):
            print(f"Train Step {step}/{len(train_loader)}")
            print(LINE_UP, end=LINE_CLEAR)
    
    train_loss = running_loss / len(train_loader)
    
    print(f"Train Step {step}/{len(train_loader)} --- Train Loss: {float(train_loss.cpu())}")
    
    return train_loss


def val(net, val_loader, loss_fn, device):
    net.eval()
    with torch.no_grad(): 
        running_loss = torch.tensor([0.0]).to(device) 
        step = 0
        for img, _ in val_loader:
            img = img.to(device)
            image_noisy = add_noise(img)
            image_noisy = image_noisy.to(device)
            output = net(image_noisy).to(device)
            loss = loss_fn(output, img).to(device)
            running_loss += loss.item()
            step += 1
            
            if step < len(val_loader):
                print(f" Val Step {step}/{len(val_loader)}")
                print(LINE_UP, end=LINE_CLEAR)
            
        val_loss = running_loss / len(val_loader)
        
    print(f"Val Step {step}/{len(val_loader)} ------- Val Loss:   {float(val_loss.cpu())}")
    
    return val_loss


def test(net, test_loader, device):
    
    # raise ValueError("Forgot to change the MEAN and STD values based on the normalization chosen!")
    net.eval()
    # MEAN = torch.tensor([0.5, 0.5, 0.5]).to(device)
    # STD = torch.tensor([0.5, 0.5, 0.5]).to(device)
    
    with torch.no_grad():
        for img, _ in test_loader:
            img = img.to(device)
            noisy = add_noise(img)
            output = net(noisy).to(device)
            
            # img = img[33] * STD[:, None, None] + MEAN[:, None, None]
            # img = noisy[33]
            # img = img.permute(1, 2, 0).cpu()
            for i in range(img.shape[0]):
                fig, axs = plt.subplots(1, 3, figsize=(10, 10))
                axs[0].imshow(img[i, :, :, :].permute(1, 2, 0).cpu())
                axs[1].imshow(noisy[i, :, :, :].permute(1, 2, 0).cpu())
                axs[2].imshow(output[i, :, :, :].permute(1, 2, 0).cpu())
                plt.show()
            exit()
        