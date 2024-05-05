import torch
from torch import nn
import torch.nn.functional as F


class Model_B(nn.Module):
    def __init__(self):
        super(Model_B, self).__init__()
        
        # encoder
        self.enc1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), 
                              stride=(1, 1), padding='same', dilation=1, groups=1, 
                              bias=True, padding_mode='zeros')
        
        self.batch_enc1 = nn.BatchNorm2d(8, affine=True)
        
        self.enc2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(2, 2), 
                              stride=(2, 2), padding='valid', dilation=1, groups=1, 
                              bias=True, padding_mode='zeros')
        
        self.batch_enc2 = nn.BatchNorm2d(8, affine=True)
        
        self.enc3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3),
                              stride=(1, 1), padding='same', dilation=1, groups=1,
                              bias=True, padding_mode='zeros')
        
        self.batch_enc3 = nn.BatchNorm2d(16, affine=True)
        
        self.enc4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 2),
                              stride=(2, 2), padding='valid', dilation=1, groups=1,
                              bias=True, padding_mode='zeros')
        
        self.batch_enc4 = nn.BatchNorm2d(16, affine=True)
        
        
        # decoder 
        self.dec1 = nn.ConvTranspose2d(in_channels=16, out_channels=16, 
                                       kernel_size=(2, 2), stride=(2, 2), 
                                       padding=0, output_padding=0, 
                                       groups=1, bias=True, 
                                       dilation=1, padding_mode='zeros')
        
        self.batch_dec1 = nn.BatchNorm2d(16, affine=True)
        
        self.dec2 = nn.ConvTranspose2d(in_channels=16, out_channels=8, 
                                       kernel_size=(3, 3), stride=(1, 1), 
                                       padding=1, output_padding=0, 
                                       groups=1, bias=True, 
                                       dilation=1, padding_mode='zeros')
        
        self.batch_dec2 = nn.BatchNorm2d(8, affine=True)
        
        self.dec3 = nn.ConvTranspose2d(in_channels=8, out_channels=8, 
                                       kernel_size=(2, 2), stride=(2, 2), 
                                       padding=0, output_padding=0, 
                                       groups=1, bias=True, 
                                       dilation=1, padding_mode='zeros')
        
        self.batch_dec3 = nn.BatchNorm2d(8, affine=True)
        
        self.dec4 = nn.ConvTranspose2d(in_channels=8, out_channels=3, 
                                       kernel_size=(3, 3), stride=(1, 1), 
                                       padding=1, output_padding=0, 
                                       groups=1, bias=True, 
                                       dilation=1, padding_mode='zeros')
        
    def forward(self, x):
        x = self.enc1(x)
        
        x = self.batch_enc1(x)
        
        x = F.relu(x)
        
        x = self.enc2(x)
        
        x = self.batch_enc2(x)
        
        x = F.relu(x)
        
        x = self.enc3(x)
        
        x = self.batch_enc3(x)
        
        x = F.relu(x)
        
        x = self.enc4(x)
        
        x = self.batch_enc4(x)
        
        x = F.relu(x)
        
        x = self.dec1(x)
        
        x = self.batch_dec1(x)
        
        x = F.relu(x)
        
        x = self.dec2(x)
        
        x = self.batch_dec2(x)
        
        x = F.relu(x)
        
        x = self.dec3(x)
        
        x = self.batch_dec3(x)
        
        x = F.relu(x)
        
        x = self.dec4(x)
        
        x = F.sigmoid(x)
        
        return x
