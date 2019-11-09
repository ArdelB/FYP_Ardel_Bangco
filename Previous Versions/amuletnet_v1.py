from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np

# Outputs only feature integration prediction 
class AmuletNet(nn.Module):
    def __init__(self, original_model):
        super(AmuletNet, self).__init__()
        all_features = list(original_model.children())[:-2][0][:-1]

        
        # VGG feature extraction layers 
        self.resolutions = [256, 128, 64, 32, 16]
        self.channels = [64, 128, 256, 512, 512]
        # 256x256
        self.layer_1 = nn.Sequential(all_features[0:4])        
        # 128x128
        self.layer_2 = nn.Sequential(all_features[4:9])       
        # 64x64
        self.layer_3 = nn.Sequential(all_features[9:16])        
        # 32x32
        self.layer_4 = nn.Sequential(all_features[16:23])        
        # 16x16
        self.layer_5 = nn.Sequential(all_features[23:])
        
        # RFC
        # 1x1 convolution to learn how to combine the feature maps                        
        self.combine_concat_features_0 = self.combine_concat()
        self.combine_concat_features_1 = self.combine_concat()
        self.combine_concat_features_2 = self.combine_concat()
        self.combine_concat_features_3 = self.combine_concat()
        self.combine_concat_features_4 = self.combine_concat()
        
        self.shrinks_n_extends = self.real_rfc(self.resolutions, self.channels)
        # Shrink/Extend - 'To Level _ From Level'
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
        
        # SMP
        
       # Saliency prediction
        self.salient_predict = nn.Conv2d(in_channels=320, out_channels=1, kernel_size=1, stride=1)
        
    def forward(self, x):
        # These will produce feature maps of N x C x H x W
        out_1 = self.layer_1(x)
        out_2 = self.layer_2(out_1)
        out_3 = self.layer_3(out_2)
        out_4 = self.layer_4(out_3)
        out_5 = self.layer_5(out_4)

        # Level 0
        F_tau_0 = self.change_0_0(out_1)
        F_tau_1 = self.change_0_1(out_2)
        F_tau_2 = self.change_0_2(out_3)
        F_tau_3 = self.change_0_3(out_4)
        F_tau_4 = self.change_0_4(out_5)
        
        rfc_out_0 = torch.cat((F_tau_0, F_tau_1, F_tau_2, F_tau_3, F_tau_4), 1)
        
        rfc_out_0 = self.combine_concat_features_0(rfc_out_0)
        
        rfc_out_0 = self.salient_predict(rfc_out_0)
        
        
        # Level 1
        #F_tau_0 = self.change_1_0(out_1)
        #F_tau_1 = self.change_1_1(out_2)
        #F_tau_2 = self.change_1_2(out_3)
        #F_tau_3 = self.change_1_3(out_4)
        #F_tau_4 = self.change_1_4(out_5)
        
        #rfc_out_1 = torch.cat((F_tau_0, F_tau_1, F_tau_2, F_tau_3, F_tau_4), 1)
        #rfc_out_1 = self.combine_concat_features_1(rfc_out_1)        
        #rfc_out_1 = self.salient_predict(rfc_out_1)
        
        # Level 2
        #F_tau_0 = self.change_2_0(out_1)
        #F_tau_1 = self.change_2_1(out_2)
        #F_tau_2 = self.change_2_2(out_3)
        #F_tau_3 = self.change_2_3(out_4)
        #F_tau_4 = self.change_2_4(out_5)
        
        #rfc_out_2 = torch.cat((F_tau_0, F_tau_1, F_tau_2, F_tau_3, F_tau_4), 1)
        #rfc_out_2 = self.combine_concat_features_2(rfc_out_2)        
        #rfc_out_2 = self.salient_predict(rfc_out_2)
        
        # Level 3
        #F_tau_0 = self.change_3_0(out_1)
        #F_tau_1 = self.change_3_1(out_2)
        #F_tau_2 = self.change_3_2(out_3)
        #F_tau_3 = self.change_3_3(out_4)
        #F_tau_4 = self.change_3_4(out_5)
        
        #rfc_out_3 = torch.cat((F_tau_0, F_tau_1, F_tau_2, F_tau_3, F_tau_4), 1)
        #rfc_out_3 = self.combine_concat_features_3(rfc_out_3)
        
        #rfc_out_3 = self.salient_predict(rfc_out_3)
        
        # Level 4
        #F_tau_0 = self.change_4_0(out_1)
        #F_tau_1 = self.change_4_1(out_2)
        #F_tau_2 = self.change_4_2(out_3)
        #F_tau_3 = self.change_4_3(out_4)
        #F_tau_4 = self.change_4_4(out_5)
        
        #rfc_out_4 = torch.cat((F_tau_0, F_tau_1, F_tau_2, F_tau_3, F_tau_4), 1)
        #rfc_out_4 = self.combine_concat_features_4(rfc_out_4)
        
        #rfc_out_4 = self.salient_predict(rfc_out_4)
        
        
        
        return rfc_out_0# , rfc_out_1, rfc_out_2, rfc_out_3]# rfc_out_4]
          
    
    def real_rfc(self, res, channels):
      L = 5
      rfc_out = [ [], [], [], [], [] ]
      
      # l will be the target resolution
      # lower l will be higher res 
      for l in range(L):
        for r in range(L):
          # if img has larger res than target res
          if r <= l:
            rfc_out[l].append(self.create_shrink(channels[r], 2**np.absolute(l-r)))
          else:
            rfc_out[l].append(self.create_extend(channels[r], 2**np.absolute(l-r)))
            
     
      return rfc_out
        
    def create_shrink(self, ci, n):
      n = int(np.floor(n))      
      if n ==1:
        K = n
        p = 0
      else:
        K = 3
        p = 1
      return nn.Sequential(
          nn.Conv2d(in_channels=ci, out_channels=64, kernel_size=K, stride=n, padding=p),
          nn.BatchNorm2d(64),
          nn.ReLU()    
      )
    
    def create_extend(self, ci, m):
      m = int(np.floor(m))      
      return nn.Sequential(
          nn.ConvTranspose2d(in_channels=ci, out_channels=64, kernel_size=3, stride=m, padding=1, output_padding=m-1),
          nn.BatchNorm2d(64),
          nn.ReLU()  
      )
    
    def combine_concat(self):
      blk = nn.Sequential(            
            nn.Conv2d(in_channels=320, out_channels=320, kernel_size=1, stride=1),
            nn.BatchNorm2d(320),
            nn.ReLU()            
            )
      return blk