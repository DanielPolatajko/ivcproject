"""Structure partially borrowed from MLP Coursework 2"""

import torch

import torch.nn as nn
import torch.nn.functional as F


"""
    Classes for convolutional ResNets of varying depth, used in main.py
    to perform deep learning on DAVIS data
"""



class ShallowNetwork(nn.Module):
    def __init__(self,input_shape):

        super(ShallowNetwork, self).__init__()

        # set up class attributes useful in building the network and inference
        self.input_shape=input_shape
        self.num_output_classes = 2

        # size 3 kernels throughout, pad to ensure shape consistency
        self.kernel_size = 3
        self.padding =  (self.kernel_size // 2, self.kernel_size // 2)

        # build the network
        self.build_module()

    def build_module(self):
        """
        Builds network whilst automatically inferring shapes of layers.

        """
        self.layer_dict = nn.ModuleDict()
        x = torch.zeros(self.input_shape, requires_grad=True)
        out = x
        self.layer_dict['conv_0'] = nn.Conv2d(in_channels=self.input_shape[1],out_channels=16,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_0'].forward(out)
        out = F.leaky_relu(out)

        temp = out
        self.layer_dict['conv_1'] = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_1'].forward(out) +temp
        out = F.leaky_relu(out)

        self.layer_dict['conv_t_2'] = nn.Conv2d(in_channels=16,out_channels=8,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_t_2'].forward(out)
        out = F.leaky_relu(out)

        temp = out
        self.layer_dict['conv_3'] = nn.Conv2d(in_channels=8,out_channels=8,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_3'].forward(out)
        out = F.leaky_relu(out) + temp

        self.layer_dict['conv_4'] = nn.Conv2d(in_channels=8,out_channels=4,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_4'].forward(out)
        out = F.leaky_relu(out)

        temp = out
        self.layer_dict['conv_5'] = nn.Conv2d(in_channels=4,out_channels=4,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_5'].forward(out) +temp
        out = F.leaky_relu(out)


        self.layer_dict['conv_t_0'] = nn.ConvTranspose2d(in_channels=4,out_channels=8,kernel_size=self.kernel_size, padding = self.padding)
        out =self.layer_dict['conv_t_0'].forward(out)
        out = F.leaky_relu(out)
        self.layer_dict['conv_t_1'] = nn.ConvTranspose2d(in_channels=8,out_channels=16,kernel_size=self.kernel_size, padding = self.padding)
        out =self.layer_dict['conv_t_1'].forward(out)
        out = F.leaky_relu(out)

        self.layer_dict['final'] = nn.Conv2d(in_channels=16,out_channels=1,kernel_size=self.kernel_size, padding = self.padding)
        out =self.layer_dict['final'].forward(out)


        print(out.shape)
        print("Block is built, output volume is", out.shape)


        return out

    def forward(self, x):

        """
            forward pass of convolutional network
        """
        out = x
        out = self.layer_dict['conv_0'].forward(out)
        out = F.leaky_relu(out)

        temp = out
        out = self.layer_dict['conv_1'].forward(out) +temp
        out = F.leaky_relu(out)

        out = self.layer_dict['conv_t_2'].forward(out)
        out = F.leaky_relu(out)

        temp = out
        out = self.layer_dict['conv_3'].forward(out)
        out = F.leaky_relu(out) + temp

        out = self.layer_dict['conv_4'].forward(out)
        out = F.leaky_relu(out)

        temp = out
        out = self.layer_dict['conv_5'].forward(out) +temp
        out = F.leaky_relu(out)

        out =self.layer_dict['conv_t_0'].forward(out)
        out = F.leaky_relu(out)

        out =self.layer_dict['conv_t_1'].forward(out)
        out = F.leaky_relu(out)

        out =self.layer_dict['final'].forward(out)

        return out

    def reset_parameters(self):
        """
        Re-initialize the network parameters.
        """
        for item in self.layer_dict.children():
            try:
                item.reset_parameters()
            except:
                pass



class DeeperNetwork(nn.Module):
    def __init__(self,input_shape):

        super(DeeperNetwork, self).__init__()

        # set up class attributes useful in building the network and inference
        self.input_shape=input_shape
        self.num_output_classes = 2
        self.kernel_size = 3
        self.padding =  (self.kernel_size // 2, self.kernel_size // 2)

        # build the network
        self.build_module()

    def build_module(self):
        """
        Builds network whilst automatically inferring shapes of layers.

        #conv2d        3, 16, 8
        #conv2d        16, 8, 8
        #transpose     8, 16, 8
        #transpose     16, 1, 8

        """
        self.layer_dict = nn.ModuleDict()
        x = torch.zeros(self.input_shape, requires_grad=True)
        out = x
        self.layer_dict['conv_0'] = nn.Conv2d(in_channels=self.input_shape[1],out_channels=32,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_0'].forward(out)
        out = F.leaky_relu(out)

        temp = out
        self.layer_dict['conv_1'] = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_1'].forward(out)
        out = F.leaky_relu(out)
        self.layer_dict['conv_2'] = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_2'].forward(out) +temp
        out = F.leaky_relu(out)

        self.layer_dict['conv_3'] = nn.Conv2d(in_channels=32,out_channels=16,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_3'].forward(out)
        out = F.leaky_relu(out)

        temp = out
        self.layer_dict['conv_4'] = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_4'].forward(out)
        out = F.leaky_relu(out)
        self.layer_dict['conv_5'] = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_5'].forward(out) +temp
        out = F.leaky_relu(out)

        temp = out
        self.layer_dict['conv_6'] = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_6'].forward(out)
        out = F.leaky_relu(out)
        self.layer_dict['conv_7'] = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_7'].forward(out) +temp
        out = F.leaky_relu(out)

        self.layer_dict['conv_8'] = nn.Conv2d(in_channels=16,out_channels=8,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_8'].forward(out)
        out = F.leaky_relu(out)

        temp = out
        self.layer_dict['conv_9'] = nn.Conv2d(in_channels=8,out_channels=8,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_9'].forward(out)
        out = F.leaky_relu(out)
        self.layer_dict['conv_10'] = nn.Conv2d(in_channels=8,out_channels=8,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_10'].forward(out) +temp
        out = F.leaky_relu(out)

        temp = out
        self.layer_dict['conv_11'] = nn.Conv2d(in_channels=8,out_channels=8,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_11'].forward(out)
        out = F.leaky_relu(out)
        self.layer_dict['conv_12'] = nn.Conv2d(in_channels=8,out_channels=8,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_12'].forward(out) +temp
        out = F.leaky_relu(out)

        self.layer_dict['conv_13'] = nn.Conv2d(in_channels=8,out_channels=4,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_13'].forward(out)
        out = F.leaky_relu(out)

        temp = out
        self.layer_dict['conv_14'] = nn.Conv2d(in_channels=4,out_channels=4,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_14'].forward(out)
        out = F.leaky_relu(out)
        self.layer_dict['conv_15'] = nn.Conv2d(in_channels=4,out_channels=4,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_15'].forward(out) +temp
        out = F.leaky_relu(out)

        temp = out
        self.layer_dict['conv_16'] = nn.Conv2d(in_channels=4,out_channels=4,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_16'].forward(out)
        out = F.leaky_relu(out)
        self.layer_dict['conv_17'] = nn.Conv2d(in_channels=4,out_channels=4,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_17'].forward(out) +temp
        out = F.leaky_relu(out)

        self.layer_dict['conv_t_0'] = nn.ConvTranspose2d(in_channels=4,out_channels=8,kernel_size=self.kernel_size, padding = self.padding)
        out =self.layer_dict['conv_t_0'].forward(out)
        out = F.leaky_relu(out)
        self.layer_dict['conv_t_1'] = nn.ConvTranspose2d(in_channels=8,out_channels=16,kernel_size=self.kernel_size, padding = self.padding)
        out =self.layer_dict['conv_t_1'].forward(out)
        out = F.leaky_relu(out)
        self.layer_dict['conv_t_2'] = nn.ConvTranspose2d(in_channels=16,out_channels=32,kernel_size=self.kernel_size, padding = self.padding)
        out =self.layer_dict['conv_t_2'].forward(out)
        out = F.leaky_relu(out)

        self.layer_dict['final'] = nn.Conv2d(in_channels=32,out_channels=1,kernel_size=self.kernel_size, padding = self.padding)
        out =self.layer_dict['final'].forward(out)


        print(out.shape)
        print("Block is built, output volume is", out.shape)


        return out

    def forward(self, x):
        out = x

        out = self.layer_dict['conv_0'].forward(out)
        out = F.leaky_relu(out)

        temp = out

        out = self.layer_dict['conv_1'].forward(out)
        out = F.leaky_relu(out)

        out = self.layer_dict['conv_2'].forward(out) +temp
        out = F.leaky_relu(out)


        out = self.layer_dict['conv_3'].forward(out)
        out = F.leaky_relu(out)

        temp = out

        out = self.layer_dict['conv_4'].forward(out)
        out = F.leaky_relu(out)

        out = self.layer_dict['conv_5'].forward(out) +temp
        out = F.leaky_relu(out)

        temp = out

        out = self.layer_dict['conv_6'].forward(out)
        out = F.leaky_relu(out)

        out = self.layer_dict['conv_7'].forward(out) +temp
        out = F.leaky_relu(out)


        out = self.layer_dict['conv_8'].forward(out)
        out = F.leaky_relu(out)

        temp = out

        out = self.layer_dict['conv_9'].forward(out)
        out = F.leaky_relu(out)

        out = self.layer_dict['conv_10'].forward(out) +temp
        out = F.leaky_relu(out)

        temp = out

        out = self.layer_dict['conv_11'].forward(out)
        out = F.leaky_relu(out)

        out = self.layer_dict['conv_12'].forward(out) +temp
        out = F.leaky_relu(out)


        out = self.layer_dict['conv_13'].forward(out)
        out = F.leaky_relu(out)

        temp = out

        out = self.layer_dict['conv_14'].forward(out)
        out = F.leaky_relu(out)

        out = self.layer_dict['conv_15'].forward(out) +temp
        out = F.leaky_relu(out)

        temp = out

        out = self.layer_dict['conv_16'].forward(out)
        out = F.leaky_relu(out)

        out = self.layer_dict['conv_17'].forward(out) +temp
        out = F.leaky_relu(out)


        out =self.layer_dict['conv_t_0'].forward(out)
        out = F.leaky_relu(out)

        out =self.layer_dict['conv_t_1'].forward(out)
        out = F.leaky_relu(out)

        out =self.layer_dict['conv_t_2'].forward(out)
        out = F.leaky_relu(out)


        out =self.layer_dict['final'].forward(out)
        return out

    def reset_parameters(self):
        """
        Re-initialize the network parameters.
        """
        for item in self.layer_dict.children():
            try:
                item.reset_parameters()
            except:
                pass



class DeepestNetwork(nn.Module):
    def __init__(self,input_shape):

        super(DeepestNetwork, self).__init__()

        # set up class attributes useful in building the network and inference
        self.input_shape=input_shape
        self.num_output_classes = 2
        self.kernel_size = 3
        self.padding =  (self.kernel_size // 2, self.kernel_size // 2)

        # build the network
        self.build_module()

    def build_module(self):
        """
        Builds network whilst automatically inferring shapes of layers.

        #conv2d        3, 16, 8
        #conv2d        16, 8, 8
        #transpose     8, 16, 8
        #transpose     16, 1, 8

        """
        self.layer_dict = nn.ModuleDict()
        x = torch.zeros(self.input_shape, requires_grad=True)
        out = x
        self.layer_dict['conv_0'] = nn.Conv2d(in_channels=self.input_shape[1],out_channels=64,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_0'].forward(out)
        out = F.leaky_relu(out)

        temp = out
        self.layer_dict['conv_1'] = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_1'].forward(out)
        out = F.leaky_relu(out)
        self.layer_dict['conv_2'] = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_2'].forward(out) +temp
        out = F.leaky_relu(out)

        temp = out
        self.layer_dict['conv_3'] = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_3'].forward(out)
        out = F.leaky_relu(out)
        self.layer_dict['conv_4'] = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_4'].forward(out) +temp
        out = F.leaky_relu(out)

        self.layer_dict['conv_5'] = nn.Conv2d(in_channels=64,out_channels=32,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_5'].forward(out)
        out = F.leaky_relu(out)

        temp = out
        self.layer_dict['conv_6'] = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_6'].forward(out)
        out = F.leaky_relu(out)
        self.layer_dict['conv_7'] = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_7'].forward(out) +temp
        out = F.leaky_relu(out)

        temp = out
        self.layer_dict['conv_8'] = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_8'].forward(out)
        out = F.leaky_relu(out)
        self.layer_dict['conv_9'] = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_9'].forward(out) +temp
        out = F.leaky_relu(out)

        temp = out
        self.layer_dict['conv_10'] = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_10'].forward(out)
        out = F.leaky_relu(out)
        self.layer_dict['conv_11'] = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_11'].forward(out) +temp
        out = F.leaky_relu(out)

        self.layer_dict['conv_12'] = nn.Conv2d(in_channels=32,out_channels=16,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_12'].forward(out)
        out = F.leaky_relu(out)

        temp = out
        self.layer_dict['conv_13'] = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_13'].forward(out)
        out = F.leaky_relu(out)
        self.layer_dict['conv_14'] = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_14'].forward(out) +temp
        out = F.leaky_relu(out)

        temp = out
        self.layer_dict['conv_15'] = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_15'].forward(out)
        out = F.leaky_relu(out)
        self.layer_dict['conv_16'] = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_16'].forward(out) +temp
        out = F.leaky_relu(out)

        temp = out
        self.layer_dict['conv_17'] = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_17'].forward(out)
        out = F.leaky_relu(out)
        self.layer_dict['conv_18'] = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_18'].forward(out) +temp
        out = F.leaky_relu(out)

        self.layer_dict['conv_19'] = nn.Conv2d(in_channels=16,out_channels=8,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_19'].forward(out)
        out = F.leaky_relu(out)

        temp = out
        self.layer_dict['conv_20'] = nn.Conv2d(in_channels=8,out_channels=8,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_20'].forward(out)
        out = F.leaky_relu(out)
        self.layer_dict['conv_21'] = nn.Conv2d(in_channels=8,out_channels=8,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_21'].forward(out) +temp
        out = F.leaky_relu(out)

        temp = out
        self.layer_dict['conv_22'] = nn.Conv2d(in_channels=8,out_channels=8,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_22'].forward(out)
        out = F.leaky_relu(out)
        self.layer_dict['conv_23'] = nn.Conv2d(in_channels=8,out_channels=8,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_23'].forward(out) +temp
        out = F.leaky_relu(out)

        temp = out
        self.layer_dict['conv_24'] = nn.Conv2d(in_channels=8,out_channels=8,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_24'].forward(out)
        out = F.leaky_relu(out)
        self.layer_dict['conv_25'] = nn.Conv2d(in_channels=8,out_channels=8,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_25'].forward(out) +temp
        out = F.leaky_relu(out)

        self.layer_dict['conv_26'] = nn.Conv2d(in_channels=8,out_channels=4,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_26'].forward(out)
        out = F.leaky_relu(out)

        temp = out
        self.layer_dict['conv_27'] = nn.Conv2d(in_channels=4,out_channels=4,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_27'].forward(out)
        out = F.leaky_relu(out)
        self.layer_dict['conv_28'] = nn.Conv2d(in_channels=4,out_channels=4,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_28'].forward(out) +temp
        out = F.leaky_relu(out)

        temp = out
        self.layer_dict['conv_29'] = nn.Conv2d(in_channels=4,out_channels=4,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_29'].forward(out)
        out = F.leaky_relu(out)
        self.layer_dict['conv_30'] = nn.Conv2d(in_channels=4,out_channels=4,kernel_size=self.kernel_size, padding = self.padding)
        out = self.layer_dict['conv_30'].forward(out) +temp
        out = F.leaky_relu(out)

        self.layer_dict['conv_t_0'] = nn.ConvTranspose2d(in_channels=4,out_channels=8,kernel_size=self.kernel_size, padding = self.padding)
        out =self.layer_dict['conv_t_0'].forward(out)
        out = F.leaky_relu(out)
        self.layer_dict['conv_t_1'] = nn.ConvTranspose2d(in_channels=8,out_channels=16,kernel_size=self.kernel_size, padding = self.padding)
        out =self.layer_dict['conv_t_1'].forward(out)
        out = F.leaky_relu(out)
        self.layer_dict['conv_t_2'] = nn.ConvTranspose2d(in_channels=16,out_channels=32,kernel_size=self.kernel_size, padding = self.padding)
        out =self.layer_dict['conv_t_2'].forward(out)
        out = F.leaky_relu(out)
        self.layer_dict['conv_t_3'] = nn.ConvTranspose2d(in_channels=32,out_channels=64,kernel_size=self.kernel_size, padding = self.padding)
        out =self.layer_dict['conv_t_3'].forward(out)
        out = F.leaky_relu(out)

        self.layer_dict['final'] = nn.Conv2d(in_channels=64,out_channels=1,kernel_size=self.kernel_size, padding = self.padding)
        out =self.layer_dict['final'].forward(out)


        print(out.shape)
        print("Block is built, output volume is", out.shape)


        return out

    def forward(self, x):
        out = x

        out = self.layer_dict['conv_0'].forward(out)
        out = F.leaky_relu(out)

        temp = out

        out = self.layer_dict['conv_1'].forward(out)
        out = F.leaky_relu(out)

        out = self.layer_dict['conv_2'].forward(out) +temp
        out = F.leaky_relu(out)

        temp = out

        out = self.layer_dict['conv_3'].forward(out)
        out = F.leaky_relu(out)

        out = self.layer_dict['conv_4'].forward(out) +temp
        out = F.leaky_relu(out)


        out = self.layer_dict['conv_5'].forward(out)
        out = F.leaky_relu(out)

        temp = out

        out = self.layer_dict['conv_6'].forward(out)
        out = F.leaky_relu(out)

        out = self.layer_dict['conv_7'].forward(out) +temp
        out = F.leaky_relu(out)

        temp = out

        out = self.layer_dict['conv_8'].forward(out)
        out = F.leaky_relu(out)

        out = self.layer_dict['conv_9'].forward(out) +temp
        out = F.leaky_relu(out)

        temp = out

        out = self.layer_dict['conv_10'].forward(out)
        out = F.leaky_relu(out)

        out = self.layer_dict['conv_11'].forward(out) +temp
        out = F.leaky_relu(out)


        out = self.layer_dict['conv_12'].forward(out)
        out = F.leaky_relu(out)

        temp = out

        out = self.layer_dict['conv_13'].forward(out)
        out = F.leaky_relu(out)

        out = self.layer_dict['conv_14'].forward(out) +temp
        out = F.leaky_relu(out)

        temp = out

        out = self.layer_dict['conv_15'].forward(out)
        out = F.leaky_relu(out)

        out = self.layer_dict['conv_16'].forward(out) +temp
        out = F.leaky_relu(out)

        temp = out

        out = self.layer_dict['conv_17'].forward(out)
        out = F.leaky_relu(out)

        out = self.layer_dict['conv_18'].forward(out) +temp
        out = F.leaky_relu(out)


        out = self.layer_dict['conv_19'].forward(out)
        out = F.leaky_relu(out)

        temp = out

        out = self.layer_dict['conv_20'].forward(out)
        out = F.leaky_relu(out)

        out = self.layer_dict['conv_21'].forward(out) +temp
        out = F.leaky_relu(out)

        temp = out

        out = self.layer_dict['conv_22'].forward(out)
        out = F.leaky_relu(out)

        out = self.layer_dict['conv_23'].forward(out) +temp
        out = F.leaky_relu(out)

        temp = out

        out = self.layer_dict['conv_24'].forward(out)
        out = F.leaky_relu(out)

        out = self.layer_dict['conv_25'].forward(out) +temp
        out = F.leaky_relu(out)


        out = self.layer_dict['conv_26'].forward(out)
        out = F.leaky_relu(out)

        temp = out

        out = self.layer_dict['conv_27'].forward(out)
        out = F.leaky_relu(out)

        out = self.layer_dict['conv_28'].forward(out) +temp
        out = F.leaky_relu(out)

        temp = out

        out = self.layer_dict['conv_29'].forward(out)
        out = F.leaky_relu(out)

        out = self.layer_dict['conv_30'].forward(out) +temp
        out = F.leaky_relu(out)


        out =self.layer_dict['conv_t_0'].forward(out)
        out = F.leaky_relu(out)

        out =self.layer_dict['conv_t_1'].forward(out)
        out = F.leaky_relu(out)

        out =self.layer_dict['conv_t_2'].forward(out)
        out = F.leaky_relu(out)

        out =self.layer_dict['conv_t_3'].forward(out)
        out = F.leaky_relu(out)

        out =self.layer_dict['final'].forward(out)
        return out

    def reset_parameters(self):
        """
        Re-initialize the network parameters.
        """
        for item in self.layer_dict.children():
            try:
                item.reset_parameters()
            except:
                pass
