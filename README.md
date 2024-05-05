[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-360/)

# Denoising
This project focuses on developing a denoising fully CNN Auto-encoder based architecture, on the CIFAR-10 dataset. Different architectures are examined and finally two of them are tested using a Visual Transformer fine tuned on the CIFAR-10 dataset. More details about this work can be found on the `Denoise.pdf` file.

## Files
In the `source/main.py` file one can train the model based on the specifications provided by the `tests_fully.csv`. The noise is added using the `source/noisy.py` file. The architectures developed can be found on  `source/model_b.py` and `source/model_c.py`. Model-C is the best model developed so far, but it can be made even better. All progress steps are logged on the `Notes_on_training.odt` file, so anyone can have an overview over the whole process. The best model checkpoints (test ids: 35, 37 and 38 as refered in `tests_fully.csv`) can be found on the `archive` folder. All test results, that is Loss and PSNR plots can be found there as well, but only for the three best models there exist the .pt pytorch model checkpoints.

## Results

## Noise

## Specifications
Processor: Intel® CoreTM i7-8550U, 4 GHz (turbo mode)
GPU: Nvidia MX150, 2 GB
RAM: 16 GB
Operating System: Ubuntu 23.10

## Contact
e-mail: chrispsyc@yahoo.com
