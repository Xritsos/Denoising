import torch



def PSNR(image, noisy_image):
    mse = torch.mean((image - noisy_image) ** 2).to(image.device)
    
    if mse == 0:
        return 0
    
    max_pixel = torch.max(image).to(image.device)
    
    psnr = (20 * torch.log10(max_pixel / torch.sqrt(mse))).to(image.device)
    
    return psnr
