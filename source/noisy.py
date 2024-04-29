import torch
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from photutils.datasets import apply_poisson_noise


def add_gaussian(img, mean=0, sigma=0.3):
	torch.manual_seed(7)
 
	noisy = torch.clone(img)

	noise = torch.normal(mean=mean, std=sigma, size=noisy.shape).to(img.device)
	noisy = noisy + noise
 
	noisy = torch.clip(noisy, 0., 1.)

	return noisy


def add_poisson(img):
    torch.manual_seed(7)
    
    noisy = torch.clone(img)
    noisy = noisy.permute((1, 2, 0)) * 255.
    noisy = torch.from_numpy(apply_poisson_noise(noisy.cpu(), seed=7)).to(img.device) / 255.
    
    noisy = noisy.permute((2, 0, 1))
    noisy = torch.clip(noisy, 0., 1.)
    
    return noisy


def add_multiplicative(img, mean=0, sigma=0.3):
	torch.manual_seed(7)

	noisy = torch.clone(img)
	noise = torch.normal(mean=mean, std=sigma, size=noisy.shape).to(img.device)
 
	noisy = noisy + noisy * noise
	noisy = torch.clip(noisy, 0., 1.)

	return noisy


def add_noise(batch):
	torch.manual_seed(7)
 
	noisy_batch = torch.clone(batch)
 
	for i in range(noisy_batch.shape[0]):
		inputs = batch[i, :, :, :]
		index = np.random.randint(low=1, high=3, size=1)
 
		if index == 1:
			noisy = add_multiplicative(inputs, mean=0.5, sigma=0.2)
		elif index == 2:
			noisy = add_gaussian(inputs, mean=0.2, sigma=0.2)
		elif index == 3:
			noisy = add_poisson(inputs)
   
		noisy_batch[i, :, :, :] = noisy
  
	return noisy_batch
