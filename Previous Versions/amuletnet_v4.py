from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np
# Only foreground excitation map but improved upsampling
class AmuletNet(nn.Module):
    def __init__(self, original_model):
        super(AmuletNet, self).__init__()
        all_features = list(original_model.children())[:-2][0][:-1]
        # checkpoint8k_fullv2
        target_res_channels = 64
        integrated_res_channel = target_res_channels * 5
        SMP_out_channel = 2
        resolutions = [256, 128, 64, 32, 16]
        channels = [64, 128, 256, 512, 512]
        
        # Feature Extraction
        # Level 0 
        # 256x256
        self.layer_1 = nn.Sequential(all_features[0:4]) 
        # Level 1
        # 128x128
        self.layer_2 = nn.Sequential(all_features[4:9])  
        # Level 2
        # 64x64
        self.layer_3 = nn.Sequential(all_features[9:16]) 
        # Level 3
        # 32x32
        self.layer_4 = nn.Sequential(all_features[16:23])  
        # Level 4
        # 16x16
        self.layer_5 = nn.Sequential(all_features[23:])
        
        # RFC
        # Get all convs/deconvs
        self.shrinks_n_extends = self.real_rfc(target_res_channels, channels)
        # Shrink/Extend - 'To Level _ From Level _'
        # Level 0
        self.change_0_0 = self.shrinks_n_extends[0][0]
        self.change_0_1 = self.shrinks_n_extends[0][1]
        self.change_0_2 = self.shrinks_n_extends[0][2]
        self.change_0_3 = self.shrinks_n_extends[0][3]
        self.change_0_4 = self.shrinks_n_extends[0][4]
        
        # Level 1 
        self.change_1_0 = self.shrinks_n_extends[1][0]
        self.change_1_1 = self.shrinks_n_extends[1][1]
        self.change_1_2 = self.shrinks_n_extends[1][2]
        self.change_1_3 = self.shrinks_n_extends[1][3]
        self.change_1_4 = self.shrinks_n_extends[1][4]
        
        # Level 2
        self.change_2_0 = self.shrinks_n_extends[2][0]
        self.change_2_1 = self.shrinks_n_extends[2][1]
        self.change_2_2 = self.shrinks_n_extends[2][2]
        self.change_2_3 = self.shrinks_n_extends[2][3]
        self.change_2_4 = self.shrinks_n_extends[2][4]
        
        # Level 3
        self.change_3_0 = self.shrinks_n_extends[3][0]
        self.change_3_1 = self.shrinks_n_extends[3][1]
        self.change_3_2 = self.shrinks_n_extends[3][2]
        self.change_3_3 = self.shrinks_n_extends[3][3]
        self.change_3_4 = self.shrinks_n_extends[3][4]
        
        # Level 4
        self.change_4_0 = self.shrinks_n_extends[4][0]
        self.change_4_1 = self.shrinks_n_extends[4][1]
        self.change_4_2 = self.shrinks_n_extends[4][2]
        self.change_4_3 = self.shrinks_n_extends[4][3]
        self.change_4_4 = self.shrinks_n_extends[4][4]
        
        # 1x1 convolution to learn how to combine the feature maps                        
        self.combine_concat_features_0 = self.combine_concat(integrated_res_channel, integrated_res_channel)
        self.combine_concat_features_1 = self.combine_concat(integrated_res_channel, integrated_res_channel)
        self.combine_concat_features_2 = self.combine_concat(integrated_res_channel, integrated_res_channel)
        self.combine_concat_features_3 = self.combine_concat(integrated_res_channel, integrated_res_channel)
        self.combine_concat_features_4 = self.combine_concat(integrated_res_channel, integrated_res_channel)
        
        
        # SMP - input to SMP is rfc_out_channel x target_res - output
        # Highest level prediction
        # SMP - deconv 
        self.SMP_deconv_4 = self.create_extend(integrated_res_channel, SMP_out_channel, 2**4)
        self.SMP_deconv_3 = self.create_extend(integrated_res_channel, SMP_out_channel, 2**3)
        self.SMP_deconv_2 = self.create_extend(integrated_res_channel, SMP_out_channel, 2**2)
        self.SMP_deconv_1 = self.create_extend(integrated_res_channel, SMP_out_channel, 2**1)
        self.SMP_deconv_0 = self.create_extend(integrated_res_channel, SMP_out_channel, 2**0)
        
        # SMP - l+1 recursive component
        self.SMP_conv_3 = nn.Conv2d(in_channels = SMP_out_channel, out_channels = SMP_out_channel, kernel_size=1,stride=1)
        self.SMP_conv_2 = nn.Conv2d(in_channels = SMP_out_channel, out_channels = SMP_out_channel, kernel_size=1,stride=1)
        self.SMP_conv_1 = nn.Conv2d(in_channels = SMP_out_channel, out_channels = SMP_out_channel, kernel_size=1,stride=1)
        self.SMP_conv_0 = nn.Conv2d(in_channels = SMP_out_channel, out_channels = SMP_out_channel, kernel_size=1,stride=1)
        
        # SMP - Learning to combine recursive component 
        self.SMP_combine_3 = nn.Conv2d(in_channels = SMP_out_channel, out_channels = SMP_out_channel, kernel_size=1,stride=1)
        self.SMP_combine_2 = nn.Conv2d(in_channels = SMP_out_channel, out_channels = SMP_out_channel, kernel_size=1,stride=1)
        self.SMP_combine_1 = nn.Conv2d(in_channels = SMP_out_channel, out_channels = SMP_out_channel, kernel_size=1,stride=1)
        self.SMP_combine_0 = nn.Conv2d(in_channels = SMP_out_channel, out_channels = SMP_out_channel, kernel_size=1,stride=1)
        
        # Boundary Refinement 
        self.br = nn.Conv2d(in_channels=channels[0], out_channels=SMP_out_channel, kernel_size=1,stride=1)
        
        # Saliency Predictions  
        self.prediction_0 =  nn.Conv2d(in_channels=SMP_out_channel, out_channels=1, kernel_size=1, stride=1)
        self.prediction_1 =  nn.Conv2d(in_channels=SMP_out_channel, out_channels=1, kernel_size=1, stride=1)
        self.prediction_2 =  nn.Conv2d(in_channels=SMP_out_channel, out_channels=1, kernel_size=1, stride=1)
        self.prediction_3 =  nn.Conv2d(in_channels=SMP_out_channel, out_channels=1, kernel_size=1, stride=1)
        self.prediction_4 =  nn.Conv2d(in_channels=SMP_out_channel, out_channels=1, kernel_size=1, stride=1)
        
        # Final Saliency Prediction
        self.final_prediction = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1)
        
        # Activation Function
        self.relu = nn.ReLU()

        
    def forward(self, x):
        # Feature Extraction
        out_1 = self.layer_1(x)
        out_2 = self.layer_2(out_1)
        out_3 = self.layer_3(out_2)
        out_4 = self.layer_4(out_3)
        out_5 = self.layer_5(out_4)
        
        # RFC
        # Level 0
        F_tau_0 = self.change_0_0(out_1)
        F_tau_1 = self.change_0_1(out_2)
        F_tau_2 = self.change_0_2(out_3)
        F_tau_3 = self.change_0_3(out_4)
        F_tau_4 = self.change_0_4(out_5)
        
        rfc_out_0 = torch.cat((F_tau_0, F_tau_1, F_tau_2, F_tau_3, F_tau_4), 1)
        
        rfc_out_0 = self.combine_concat_features_0(rfc_out_0)
        
        # Level 1
        F_tau_0 = self.change_1_0(out_1)
        F_tau_1 = self.change_1_1(out_2)
        F_tau_2 = self.change_1_2(out_3)
        F_tau_3 = self.change_1_3(out_4)
        F_tau_4 = self.change_1_4(out_5)
        
        rfc_out_1 = torch.cat((F_tau_0, F_tau_1, F_tau_2, F_tau_3, F_tau_4), 1)
        
        rfc_out_1 = self.combine_concat_features_1(rfc_out_1)
        
        # Level 2
        F_tau_0 = self.change_2_0(out_1)
        F_tau_1 = self.change_2_1(out_2)
        F_tau_2 = self.change_2_2(out_3)
        F_tau_3 = self.change_2_3(out_4)
        F_tau_4 = self.change_2_4(out_5)
        
        rfc_out_2 = torch.cat((F_tau_0, F_tau_1, F_tau_2, F_tau_3, F_tau_4), 1)
        
        rfc_out_2 = self.combine_concat_features_2(rfc_out_2)
        
        # Level 3
        F_tau_0 = self.change_3_0(out_1)
        F_tau_1 = self.change_3_1(out_2)
        F_tau_2 = self.change_3_2(out_3)
        F_tau_3 = self.change_3_3(out_4)
        F_tau_4 = self.change_3_4(out_5)
        
        rfc_out_3 = torch.cat((F_tau_0, F_tau_1, F_tau_2, F_tau_3, F_tau_4), 1)
        
        rfc_out_3 = self.combine_concat_features_3(rfc_out_3)
        
        # Level 4
        F_tau_0 = self.change_4_0(out_1)
        F_tau_1 = self.change_4_1(out_2)
        F_tau_2 = self.change_4_2(out_3)
        F_tau_3 = self.change_4_3(out_4)
        F_tau_4 = self.change_4_4(out_5)
        
        rfc_out_4 = torch.cat((F_tau_0, F_tau_1, F_tau_2, F_tau_3, F_tau_4), 1)
        
        rfc_out_4 = self.combine_concat_features_4(rfc_out_4)
        
        # SMP
        
        SMP_out_4 = self.SMP_deconv_4(rfc_out_4)
        SMP_out_3 = self.SMP_combine_3(self.relu(self.SMP_deconv_3(rfc_out_3) + self.SMP_conv_3(SMP_out_4)))
        SMP_out_2 = self.SMP_combine_2(self.relu(self.SMP_deconv_2(rfc_out_2) + self.SMP_conv_3(SMP_out_3)))
        SMP_out_1 = self.SMP_combine_1(self.relu(self.SMP_deconv_1(rfc_out_1) + self.SMP_conv_3(SMP_out_2)))
        SMP_out_0 = self.SMP_combine_0(self.relu(self.SMP_deconv_0(rfc_out_0) + self.SMP_conv_3(SMP_out_1)))
        
        prediction_0 = self.prediction_0(self.relu(SMP_out_0 + self.br(out_1)))
        prediction_1 = self.prediction_1(self.relu(SMP_out_1 + self.br(out_1)))
        prediction_2 = self.prediction_2(self.relu(SMP_out_2 + self.br(out_1)))
        prediction_3 = self.prediction_3(self.relu(SMP_out_3 + self.br(out_1)))
        prediction_4 = self.prediction_4(self.relu(SMP_out_4 + self.br(out_1)))
        
        stacked_prediction = torch.cat((prediction_0, prediction_1, prediction_2, prediction_3, prediction_4), 1)
        
        fused_prediction = self.final_prediction(stacked_prediction)
        
        return fused_prediction, [prediction_0, prediction_1, prediction_2, prediction_3, prediction_4]
          
    
    def real_rfc(self, tgt_res_channels, channels):
      L = 5
      rfc_out = [ [], [], [], [], [] ]
      
      # l will be the target resolution
      # lower l will be higher res 
      for l in range(L):
        for r in range(L):
          # if img has larger res than target res
          if r <= l:
            rfc_out[l].append(self.create_shrink(channels[r], tgt_res_channels, 2**np.absolute(l-r)))
          else:               
            rfc_out[l].append(self.create_extend(channels[r], tgt_res_channels, 2**np.absolute(l-r)))            
            
      return rfc_out
        
    def create_shrink(self, ci, co, n):
      n = int(np.floor(n))      
      if n ==1:
        K = n
        p = 0
      else:
        K = 3
        p = 1
      return nn.Sequential(
          nn.Conv2d(in_channels=ci, out_channels=co, kernel_size=K, stride=n, padding=p),
          nn.BatchNorm2d(co),
          nn.ReLU()    
      )
    
    def create_extend(self, ci, co, m):
      if m ==1:
        p = 0
        k = m
      else:
        p = int(m/2)
        k =2*m
       
      
      return nn.Sequential(
          nn.ConvTranspose2d(in_channels=ci, out_channels=co, kernel_size=k, stride=m, padding=p),
          nn.BatchNorm2d(co),
          nn.ReLU()  
      )
    
    def combine_concat(self, ci, co):
      blk = nn.Sequential(            
            nn.Conv2d(in_channels=ci, out_channels=co, kernel_size=1, stride=1),
            nn.BatchNorm2d(co),
            nn.ReLU()            
            )
      return blk