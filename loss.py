import torch
import torch.nn as nn


class MusicCompLoss(nn.Module):
    """
    Loss function utilised for implementing the IB principle.

    A combination of the MSE, with a window applied, and the Cross Entropy, with probability distributions calculated from the Fourier amplitude spectrum.

    Args:
        beta (float): The weight applied to the MSE.
        time_win ([np.array]): The window applied to the MSE. Must have len=#samples. Should be such that the edge samples are valued higher than middle; to avoid high frequency noise.
        freq_win ([np.array]): The window applied to the CE. Should be defined in the frequency domain.
    """
    
    def __init__(self, beta, time_win, freq_win):

        super(MusicCompLoss, self).__init__()
        self.beta = beta    # Scaling of entropy in loss function 
        self.time_win = time_win
        self.fft_win = freq_win
        self.softmax = nn.Softmax(dim=1)
    
    
    def forward(self, y, x):
        """
        y: output layer
        x: input layer
        q: quantized layer
        """
        ### CE Calculation ###
        # Calculates the real part of the Fourier amplitude spectrum.
        x_fft_full = torch.pow(torch.fft.rfft(x, n=len(self.time_win), dim=1).abs(), self.fft_win)
        y_fft_full = torch.pow(torch.fft.rfft(y, n=len(self.time_win), dim=1).abs(), self.fft_win)

        # Makes sure we don't have values that are 0.
        x_fft_win = torch.clamp(x_fft_full, min=1e-16) 
        y_fft_win = torch.clamp(y_fft_full, min=1e-16) 

        x_fft_s = torch.nn.functional.normalize(x_fft_win, p=1)
        y_fft = torch.nn.functional.normalize(y_fft_win, p=1)

        # The actual CE calculation
        loss_HX_CE = torch.mean(-torch.sum(torch.mul(torch.log(y_fft), x_fft_s), dim=1))


        ### MSE Calculations ###
        # Multiplies the window with the input
        x_win = torch.mul(self.time_win, x)
        y_win = torch.mul(self.time_win, y)

        loss_mse = torch.nn.functional.mse_loss(y_win, x_win, reduction='mean')


        ### Loss Function ###
        loss = loss_HX_CE + self.beta*loss_mse
        
        return loss, loss_mse, loss_HX_CE

    
class Quantizer_func(torch.autograd.Function):
    """
    The quantiser function utilised by the quantisation layer. This is simply a quantiser which discretices the input based on the parameters.

    Args:
        input (torch.Tensor): Values to be quantised.
        q_min (float): Lower range for quantisation.
        q_max (float): Upper range for quantisation.
        b_max (int): Discrete values (2^b_max)
        scale (float): b_max/(q_max - q_min)

    """

    @staticmethod
    def forward(ctx, input, q_min, q_max, b_max, scale):

        ctx.save_for_backward(torch.tensor(q_min), torch.tensor(q_max))

        # Shifts all input values up s.t. the quantization interval goes from [q_min,q_max] to [0,q_max-q_min]
        input_upshift = torch.sub(input, q_min)

        # Scales input values s.t. they quantisation interval becomes [0, b_max]
        input_scale = torch.mul(input_upshift, scale)
        
        # Rounds values to nearest integer
        input_quan = torch.round(input_scale)

        # "Clips" all values smaller than 0 to 0
        input_quan = torch.where(input_quan < 0, torch.zeros_like(input_quan), input_quan)
        
        # "Clips" all values above b_max to b_max
        input_quan = torch.where(input_quan > b_max, b_max*torch.ones_like(input_quan), input_quan)

        # Downscales all values s.t. they are contained in [0,q_max-q_min]
        quan_scale = torch.div(input_quan, scale)
        
        # Shifts all values s.t. the interval becomes [q_min,q_max]
        quan_shift = torch.add(quan_scale, q_min)

        return quan_shift
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Gradients are defined as 1 if the input lies withing [q_min, q_max], and 0 otherwise.
        """

        q_min, q_max = ctx.saved_tensors

        # Boolean arrays determinig the where clipping occours.
        inputs_below_qmin = torch.where(grad_output < q_min, torch.ones_like(grad_output), torch.zeros_like(grad_output))  # (1: <q_min | 0: >=q_min)
        inputs_above_qmax = torch.where(grad_output > q_max, torch.ones_like(grad_output), torch.zeros_like(grad_output))  # (1: >q_max | 0: =<q_max) 

        # Subtracts the boolean arrays to determine where the gradient should be 1 and 0.
        quan_grad = torch.sub(torch.sub(torch.ones_like(grad_output),inputs_below_qmin), inputs_above_qmax)

        # For some reason needs to return 5 things to work.
        return torch.mul(grad_output, quan_grad), None, None, None, None



class Quantizer(nn.Module):
    """
    The quantisation layer. This is what is called when defining the network.

    Args:
        b (int): Discrete values (2^b).
        q_int (tuple): Quantisation range.
    """

    def __init__(self, b, q_int):
        super(Quantizer, self).__init__()            
        self.q_min, self.q_max = q_int
        
        # Maximal integer value of quantised signal
        self.b_max = 2**b - 1        

        # Scaling performed on inputs
        self.scale = self.b_max/(self.q_max - self.q_min)           

    def forward(self, input):
        output = Quantizer_func.apply(input, self.q_min, self.q_max, self.b_max, self.scale).clone()
        return output