import torch
from torch import nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        # encoder
        self.enc1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), 
                              stride=(1, 1), padding=0, dilation=1, groups=1, 
                              bias=True, padding_mode='zeros')
        
        self.enc2 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(3, 3), 
                              stride=(1, 1), padding=0, dilation=1, groups=1, 
                              bias=True, padding_mode='zeros')
        
        # decoder 
        self.dec1 = nn.ConvTranspose2d(in_channels=4, out_channels=8, 
                                       kernel_size=(3, 3), stride=1, padding=0, 
                                       output_padding=0, groups=1, bias=True, 
                                       dilation=1, padding_mode='zeros')
        
        self.dec2 = nn.ConvTranspose2d(in_channels=8, out_channels=3, 
                                       kernel_size=(3, 3), stride=1, padding=0, 
                                       output_padding=0, groups=1, bias=True, 
                                       dilation=1, padding_mode='zeros')
        
    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        
        return x
