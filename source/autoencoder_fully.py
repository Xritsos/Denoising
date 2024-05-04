import torch
from torch import nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        # encoder
        self.enc1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), 
                              stride=(1, 1), padding='same', dilation=1, groups=1, 
                              bias=True, padding_mode='zeros')
        
        self.batch_enc1 = nn.BatchNorm2d(32, affine=True)
        
        self.enc2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(2, 2), 
                              stride=(2, 2), padding='valid', dilation=1, groups=1, 
                              bias=True, padding_mode='zeros')
        
        self.batch_enc2 = nn.BatchNorm2d(32, affine=True)
        
        self.enc3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3),
                              stride=(1, 1), padding='same', dilation=1, groups=1,
                              bias=True, padding_mode='zeros')
        
        self.batch_enc3 = nn.BatchNorm2d(64, affine=True)
        
        self.enc4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 2),
                              stride=(2, 2), padding='valid', dilation=1, groups=1,
                              bias=True, padding_mode='zeros')
        
        self.batch_enc4 = nn.BatchNorm2d(64, affine=True)
        
        # self.enc5 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3),
        #                       stride=(1, 1), padding='same', dilation=1, groups=1,
        #                       bias=True, padding_mode='zeros')
        
        # self.batch_enc5 = nn.BatchNorm2d(32, affine=True)
        
        # self.enc6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(2, 2),
        #                       stride=(2, 2), padding='valid', dilation=1, groups=1,
        #                       bias=True, padding_mode='zeros')
        
        # self.batch_enc6 = nn.BatchNorm2d(32, affine=True)
        
        # self.enc7 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3),
        #                       stride=(1, 1), padding='same', dilation=1, groups=1,
        #                       bias=True, padding_mode='zeros')
        
        # self.batch_enc7 = nn.BatchNorm2d(8, affine=True)
        
        # decoder 
        self.dec1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, 
                                       kernel_size=(2, 2), stride=(2, 2), 
                                       padding=0, output_padding=0, 
                                       groups=1, bias=True, 
                                       dilation=1, padding_mode='zeros')
        
        self.batch_dec1 = nn.BatchNorm2d(64, affine=True)
        
        self.dec2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, 
                                       kernel_size=(3, 3), stride=(1, 1), 
                                       padding=1, output_padding=0, 
                                       groups=1, bias=True, 
                                       dilation=1, padding_mode='zeros')
        
        self.batch_dec2 = nn.BatchNorm2d(32, affine=True)
        
        self.dec3 = nn.ConvTranspose2d(in_channels=32, out_channels=32, 
                                       kernel_size=(2, 2), stride=(2, 2), 
                                       padding=0, output_padding=0, 
                                       groups=1, bias=True, 
                                       dilation=1, padding_mode='zeros')
        
        self.batch_dec3 = nn.BatchNorm2d(32, affine=True)
        
        self.dec4 = nn.ConvTranspose2d(in_channels=32, out_channels=3, 
                                       kernel_size=(3, 3), stride=(1, 1), 
                                       padding=1, output_padding=0, 
                                       groups=1, bias=True, 
                                       dilation=1, padding_mode='zeros')
        
        # self.batch_dec4 = nn.BatchNorm2d(32, affine=True)
        
        # self.dec5 = nn.ConvTranspose2d(in_channels=32, out_channels=64, 
        #                                kernel_size=(3, 3), stride=(1, 1), 
        #                                padding=1, output_padding=0, 
        #                                groups=1, bias=True, 
        #                                dilation=1, padding_mode='zeros')
        
        # self.batch_dec5 = nn.BatchNorm2d(64, affine=True)
        
        # self.dec6 = nn.ConvTranspose2d(in_channels=64, out_channels=64, 
        #                                kernel_size=(2, 2), stride=(2, 2), 
        #                                padding=0, output_padding=0, 
        #                                groups=1, bias=True, 
        #                                dilation=1, padding_mode='zeros')
        
        # self.batch_dec6 = nn.BatchNorm2d(64, affine=True)
        
        # self.dec7 = nn.ConvTranspose2d(in_channels=64, out_channels=3, 
        #                                kernel_size=(3, 3), stride=(1, 1), 
        #                                padding=1, output_padding=0, 
        #                                groups=1, bias=True, 
        #                                dilation=1, padding_mode='zeros')
        
    def forward(self, x):
        # print(f"Input of first layer: {x.shape}")
        x = self.enc1(x)
        
        # print(f"Output of first layer: {x.shape}")
        
        x = self.batch_enc1(x)
        
        x = F.relu(x)
        
        x = self.enc2(x)
        
        x = self.batch_enc2(x)
        
        x = F.relu(x)
        
        # print(f"Output of second layer: {x.shape}")
        
        x = self.enc3(x)
        
        x = self.batch_enc3(x)
        
        x = F.relu(x)
        
        # print(f"Output of third layer: {x.shape}")
        
        x = self.enc4(x)
        
        x = self.batch_enc4(x)
        
        x = F.relu(x)
        
        # print(f"Output of fourth layer: {x.shape}")
        
        # x = self.enc5(x)
        
        # x = self.batch_enc5(x)
        
        # x = F.relu(x)
        
        # print(f"Output of fifth layer: {x.shape}")
        
        # x = self.enc6(x)
        
        # x = self.batch_enc6(x)
        
        # x = F.relu(x)
        
        # print(f"Output of sixth layer: {x.shape}")
        
        # x = self.enc7(x)
        
        # x = self.batch_enc7(x)
        
        # x = F.relu(x)
        
        # print(f"Output of seventh layer: {x.shape}")
        
        x = self.dec1(x)
        
        x = self.batch_dec1(x)
        
        x = F.relu(x)
        
        # print(f"Output of first dec layer: {x.shape}")
        
        x = self.dec2(x)
        
        x = self.batch_dec2(x)
        
        x = F.relu(x)
        
        # print(f"Output of second dec layer: {x.shape}")
        
        x = self.dec3(x)
        
        x = self.batch_dec3(x)
        
        x = F.relu(x)
        
        # print(f"Output of third dec layer: {x.shape}")
        
        x = self.dec4(x)
        
        # print(f"Output of fourth dec layer: {x.shape}")
        
        # x = F.relu(self.dec5(x))
        
        # print(f"Output of fifth dec layer: {x.shape}")
        
        # x = F.relu(self.dec6(x))
        
        # print(f"Output of sixth dec layer: {x.shape}")
        
        # x = F.sigmoid(self.dec7(x))
        
        # print(f"Output of seventh dec layer: {x.shape}")
        
        # exit()
        
        x = F.sigmoid(x)
        
        return x
