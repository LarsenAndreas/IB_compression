import torch
import torch.nn as nn
from loss import Quantizer


class NeuralNetConv(nn.Module):
    """
    The neural net definition.

        Args:
            input_size (int): The "frame" length of the input.
            b (int): Bit-Depth.
            q_interval (tuple): Quantisation interval.
            q_nodes (int): Nodes in the quantisation layer. 
            kernel_size (int): Size of all the kernels.
            channel_nums (tuple): Convolution block and channels/kernels utilized.
            padding_mode (str, optional): How to pad. Defaults to 'zeros'.
        """
    
    def __init__(self, input_size, b, q_interval, q_nodes, kernel_size, channel_nums, padding_mode='zeros'):
        
        super(NeuralNetConv, self).__init__()

        # Encoding Layers
        self.enc_convblocks = nn.ModuleList([])
        lastdim = 1
        for c in channel_nums:
            block = nn.Sequential(
                nn.Conv1d(lastdim, c, kernel_size, padding='same', padding_mode=padding_mode),
                nn.ELU()
            )
            self.enc_convblocks.append(block)
            lastdim = c 


        # Quantizing Layer
        self.quantize = nn.Sequential(
            nn.Linear(channel_nums[2]*input_size, q_nodes),
            Quantizer(b, q_interval)
        )


        #Decoding Layers
        self.dec_prep = nn.Sequential(
            nn.Linear(q_nodes, input_size),
            nn.ELU()
        )

        self.dec_convblocks = nn.ModuleList([])
        lastdim = 1
        for c in reversed(channel_nums):
            block = nn.Sequential(
                nn.Conv1d(lastdim, c, kernel_size, padding='same', padding_mode=padding_mode),
                nn.ELU()
            )
            self.dec_convblocks.append(block)
            lastdim = c
        
        self.dec_out = nn.Sequential(
            nn.Linear(channel_nums[0]*input_size, input_size),
            nn.Tanh()
        )


    def forward(self, input):

        # Encoding layers
        out = input.unsqueeze(1)  # Dimension fitting
        for block in self.enc_convblocks:
            out = block(out)  
        out = torch.flatten(out, 1)  # Dimension fitting

        # Quantisation layer
        out = self.quantize(out)

        # Decoding layers
        out = self.dec_prep(out)
        out = out.unsqueeze(1)  # Dimension fitting

        for block in self.dec_convblocks:
            out = block(out)
        out = torch.flatten(out, 1)  # Dimension fitting
        
        out = self.dec_out(out)

        return out