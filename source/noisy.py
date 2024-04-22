import torch

def add_noise(inputs,noise_factor=0.3):
    torch.manual_seed(7)
    
	noisy = inputs+torch.randn_like(inputs) * noise_factor
	noisy = torch.clip(noisy,0.,1.)
	return noisy

