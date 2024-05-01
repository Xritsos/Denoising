import torch
import numpy as np
import seaborn as sns
from random import randint
from matplotlib import pyplot as plt


def add_gaussian(img, mean, sigma):
 
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



def add_multiplicative(img, mean, sigma):

	noisy = torch.clone(img).to(img.device)
	noise = torch.normal(mean=mean, std=sigma, size=noisy.shape).to(img.device)
 
	noisy = noisy + torch.multiply(noisy, noise)
	noisy = torch.clip(noisy, 0., 1.)

	return noisy


def add_noise(batch):
 
	noisy_batch = torch.clone(batch).to(batch.device)
 
	multi_mean = torch.tensor(0.5)
	multi_sigma = torch.tensor(0.3)
	gauss_mean = torch.tensor(0.3)
	gauss_sigma = torch.tensor(0.05)
 
	for i in range(noisy_batch.shape[0]):
		inputs = batch[i, :, :, :].to(batch.device)
		
		index = randint(1, 4)
  
		if index == 1:
			noisy = add_multiplicative(inputs, mean=0.8, sigma=0.5)
		elif index == 2:
			noisy = add_gaussian(inputs, mean=0.3, sigma=0.05)
		elif index == 3:
			noisy = add_multiplicative(inputs, mean=0.2, sigma=0.2)
			noisy = add_gaussian(noisy, mean=0.1, sigma=0.08)
		else:
			noisy = add_gaussian(inputs, mean=0.2, sigma=0.05)
			noisy = add_multiplicative(noisy, mean=0.2, sigma=0.1)
   
		noisy_batch[i, :, :, :] = noisy
  
	return noisy_batch
