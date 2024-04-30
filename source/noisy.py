import torch
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


@torch.jit.script
def add_gaussian(img, mean, sigma):
	torch.manual_seed(7)
 
	noisy = torch.clone(img).to(img.device)

	noise = torch.normal(mean=mean, std=sigma, size=noisy.shape).to(img.device)
	noisy = noisy + noise
 
	noisy = torch.clip(noisy, 0., 1.)

	return noisy

# def add_poisson(img):
#     torch.manual_seed(7)
    
#     noisy = torch.clone(img).to(img.device)
#     noisy = noisy.permute((1, 2, 0)) * 255.
#     noisy = torch.from_numpy(apply_poisson_noise(noisy.cpu(), seed=7)).to(img.device) / 255.
    
#     noisy = noisy.permute((2, 0, 1))
#     noisy = torch.clip(noisy, 0., 1.)
    
#     return noisy


@torch.jit.script
def add_multiplicative(img, mean, sigma):
	torch.manual_seed(7)

	noisy = torch.clone(img).to(img.device)
	noise = torch.normal(mean=mean, std=sigma, size=noisy.shape).to(img.device)
 
	noisy = noisy + torch.multiply(noisy, noise)
	noisy = torch.clip(noisy, 0., 1.)

	return noisy


@torch.jit.script
def add_noise(batch):
	torch.manual_seed(7)
 
	noisy_batch = torch.clone(batch).to(batch.device)
 
	multi_mean = torch.tensor(1.0)
	multi_sigma = torch.tensor(0.5)
	gauss_mean = torch.tensor(0.2)
	gauss_sigma = torch.tensor(0.2)
 
	for i in range(noisy_batch.shape[0]):
		inputs = batch[i, :, :, :].to(batch.device)
		index = torch.randint(low=1, high=4, size=[1]).to(batch.device)
  
		if index == 1:
			noisy = add_multiplicative(inputs, mean=multi_mean, sigma=multi_sigma)
		elif index == 2:
			noisy = add_gaussian(inputs, mean=gauss_mean, sigma=gauss_sigma)
		elif index == 3:
			noisy = add_multiplicative(inputs, mean=multi_mean, sigma=multi_sigma)
			noisy = add_gaussian(noisy, mean=gauss_mean, sigma=gauss_sigma)
		else:
			noisy = add_gaussian(inputs, mean=gauss_mean, sigma=gauss_sigma)
			noisy = add_multiplicative(noisy, mean=multi_mean, sigma=multi_sigma)
   
		noisy_batch[i, :, :, :] = noisy
  
	return noisy_batch
