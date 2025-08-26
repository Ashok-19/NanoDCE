# model_student.py (Corrected)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class CSDN_Tem(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CSDN_Tem, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class enhance_net_nopool_student(nn.Module): # Renamed for clarity

    def __init__(self, scale_factor):
        super(enhance_net_nopool_student, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
        number_f = 4  # Keep the reduced channels from the student model


        self.e_conv1 = CSDN_Tem(3, number_f)
        self.e_conv2 = CSDN_Tem(number_f,number_f)
        self.e_conv3 = CSDN_Tem(number_f, 3)
        

    def enhance(self, x, x_r):

        out = x
        
        #LE curve
        for _ in range(4):
             out = out + x_r * (out * out - out)
        return out
 
    def forward(self, x):
        if self.scale_factor == 1:
            x_down = x
        else:
            # Downsample input if scale_factor > 1
            x_down = F.interpolate(x, scale_factor=1/self.scale_factor, mode='bilinear')

        
        x1 = self.relu(self.e_conv1(x_down))
        x2 = self.relu(self.e_conv2(x1))
        
        x_r = F.tanh(self.e_conv3(x2))
               

        # Upsample x_r back to original input size if needed
        if self.scale_factor == 1:
            x_r = x_r
        else:
            x_r = self.upsample(x_r)

        # --- Apply enhancement ---
        enhance_image = self.enhance(x, x_r)

        return enhance_image, x_r