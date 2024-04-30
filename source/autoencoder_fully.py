import torch
from torch import nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        # encoder
        self.enc1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), 
                              stride=(1, 1), padding='same', dilation=1, groups=1, 
                              bias=True, padding_mode='zeros')
        
        self.enc2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 2), 
                              stride=(2, 2), padding='valid', dilation=1, groups=1, 
                              bias=True, padding_mode='zeros')
        
        self.enc3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3),
                              stride=(1, 1), padding='same', dilation=1, groups=1,
                              bias=True, padding_mode='zeros')
        
        self.enc4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(2, 2),
                              stride=(2, 2), padding='valid', dilation=1, groups=1,
                              bias=True, padding_mode='zeros')
        
        self.enc5 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3),
                              stride=(1, 1), padding='same', dilation=1, groups=1,
                              bias=True, padding_mode='zeros')
        
        self.enc6 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 2),
                              stride=(2, 2), padding='valid', dilation=1, groups=1,
                              bias=True, padding_mode='zeros')
        
        self.enc7 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3),
                              stride=(1, 1), padding='same', dilation=1, groups=1,
                              bias=True, padding_mode='zeros')
        
        # decoder 
        self.dec1 = nn.ConvTranspose2d(in_channels=8, out_channels=16, 
                                       kernel_size=(3, 3), stride=(1, 1), 
                                       padding=1, output_padding=0, 
                                       groups=1, bias=True, 
                                       dilation=1, padding_mode='zeros')
        
        self.dec2 = nn.ConvTranspose2d(in_channels=16, out_channels=16, 
                                       kernel_size=(2, 2), stride=(2, 2), 
                                       padding=0, output_padding=0, 
                                       groups=1, bias=True, 
                                       dilation=1, padding_mode='zeros')
        
        self.dec3 = nn.ConvTranspose2d(in_channels=16, out_channels=32, 
                                       kernel_size=(3, 3), stride=(1, 1), 
                                       padding=1, output_padding=0, 
                                       groups=1, bias=True, 
                                       dilation=1, padding_mode='zeros')
        
        
        self.dec4 = nn.ConvTranspose2d(in_channels=32, out_channels=32, 
                                       kernel_size=(2, 2), stride=(2, 2), 
                                       padding=0, output_padding=0, 
                                       groups=1, bias=True, 
                                       dilation=1, padding_mode='zeros')
        
        self.dec5 = nn.ConvTranspose2d(in_channels=32, out_channels=64, 
                                       kernel_size=(3, 3), stride=(1, 1), 
                                       padding=1, output_padding=0, 
                                       groups=1, bias=True, 
                                       dilation=1, padding_mode='zeros')
        
        self.dec6 = nn.ConvTranspose2d(in_channels=64, out_channels=64, 
                                       kernel_size=(2, 2), stride=(2, 2), 
                                       padding=0, output_padding=0, 
                                       groups=1, bias=True, 
                                       dilation=1, padding_mode='zeros')
        
        self.dec7 = nn.ConvTranspose2d(in_channels=64, out_channels=3, 
                                       kernel_size=(3, 3), stride=(1, 1), 
                                       padding=1, output_padding=0, 
                                       groups=1, bias=True, 
                                       dilation=1, padding_mode='zeros')
        
    def forward(self, x):
        # print(f"Input of first layer: {x.shape}")
        x = F.relu(self.enc1(x))
        
        # print(f"Output of first layer: {x.shape}")
        
        x = F.relu(self.enc2(x))
        
        # print(f"Output of second layer: {x.shape}")
        
        x = F.relu(self.enc3(x))
        
        # print(f"Output of third layer: {x.shape}")
        
        x = F.relu(self.enc4(x))
        
        # print(f"Output of fourth layer: {x.shape}")
        
        x = F.relu(self.enc5(x))
        
        # print(f"Output of fifth layer: {x.shape}")
        
        x = F.relu(self.enc6(x))
        
        # print(f"Output of sixth layer: {x.shape}")
        
        x = F.relu(self.enc7(x))
        
        # print(f"Output of seventh layer: {x.shape}")
        
        x = F.relu(self.dec1(x))
        
        # print(f"Output of first dec layer: {x.shape}")
        
        x = F.relu(self.dec2(x))
        
        # print(f"Output of second dec layer: {x.shape}")
        
        x = F.relu(self.dec3(x))
        
        # print(f"Output of third dec layer: {x.shape}")
        
        x = F.sigmoid(self.dec4(x))
        
        # print(f"Output of fourth dec layer: {x.shape}")
        
        x = F.relu(self.dec5(x))
        
        # print(f"Output of fifth dec layer: {x.shape}")
        
        x = F.relu(self.dec6(x))
        
        # print(f"Output of sixth dec layer: {x.shape}")
        
        x = F.sigmoid(self.dec7(x))
        
        # print(f"Output of seventh dec layer: {x.shape}")
        
        # exit()
        
        return x
