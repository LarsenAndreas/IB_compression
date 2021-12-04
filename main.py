import torch
import random

import numpy as np

from tqdm import tqdm
from scipy.signal import windows
from torch.utils.data import DataLoader

# Custom packages
import net
import data
import utils
import loss


def getFreqWin():
    """
    Window used for weighing the Fourier amplitude spectrum.
    """

    win = 100*np.array([
        0.01001502, 0.02186158, 0.02468514, 0.02473119, 0.02344306,
        0.02420558, 0.02614269, 0.02733992, 0.027928  , 0.02808134,
        0.02791206, 0.02747797, 0.02683388, 0.02604171, 0.0251617 ,
        0.02424665, 0.02334555, 0.02249787, 0.02173223, 0.02106286,
        0.02048341, 0.01998594, 0.01956418, 0.01921331, 0.01892948,
        0.0187096 , 0.01855168, 0.01845486, 0.01841943, 0.01844628,
        0.01852913, 0.01865568, 0.0188135 , 0.01898964, 0.01917029,
        0.01934057, 0.01948487, 0.01959483, 0.01967077, 0.01971386,
        0.01972565, 0.019708  , 0.01966303, 0.01959306, 0.01950055,
        0.01938807, 0.0192582 , 0.01911351, 0.01895654, 0.01878973,
        0.01861543, 0.01843586, 0.01825311, 0.01806913, 0.01788572,
        0.01770456, 0.01752718, 0.017355  , 0.01718931, 0.01703132,
        0.01688213, 0.01674279, 0.01661427, 0.01649752, 0.01639344
    ])
    return win


def trainingLoop(model, dataloader, loss_func, learning_rate, n_epochs, device='cpu', desc='Default'):
    """
    The neural network training loop. This trains the autoencoder to compress the tracks.

    Args:
        model (nn.Module): The neural network description.
        dataloader (torch.Dataloader): The custom pytorch dataloader.
        loss_func (nn.Module): The loss function.
        learning_rate (float): Learning rate.
        n_epochs (int): Number of epochs.
        device (str, optional): What device does the computations. Defaults to 'cpu'.
        desc (str, optional): The name of the weights saved after each epoch. Defaults to 'Default'.
    """

    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):
        
        loss_total = 0
        mse_total = 0
        ce_total = 0

        # Creates a neat progress bar
        pbar = tqdm(total=len(dataloader), dynamic_ncols=True, desc=f'Epoch {epoch + 1}/{n_epochs}')
        for i, batch in enumerate(dataloader):
            
            # Resetting gradients
            optimizer.zero_grad()

            # Loads batch
            batch = batch.to(device)

            # Inputs batches into NN
            outputs = model(batch)

            # Calculates loss
            loss, mse, ce = loss_func(outputs, batch)

            # Backpropagation
            loss.backward()
            optimizer.step()
            
            # Log losses for progress display
            loss_total += loss.item()
            mse_total += mse.item()
            ce_total += ce.item()
            
            # Only update tqdm sometimes to reduce cpu load
            if (i + 1) % 50 == 0:
                pbar.set_postfix({'Avg Loss':f'{loss_total/(i+1):.8f}' , 'Avg MSE': f'{mse_total/(i+1):.8f}', 'Avg CE': f'{ce_total/(i+1):.8f}'})
                pbar.update(50)

        # Save model weights
        pbar.close()
        torch.save(model.state_dict(), f'weights_{desc}.pth')


if __name__ == '__main__':
    ##########################
    ### Dataset Parameters ###
    ##########################

    path = 'fma_small'              # Path to fma_small dataset (larger might also work)
    n_files = 2000                  # Number of files to be included in the training data
    duration =  1                   # Second to include from track. Set to None for entire track
    offset = 10                     # Seconds to start loading file from
    shuffle_tracks = True           # Shuffle tracks
    input_size = 2**7               # Framesize for dataloader        
                                        #The dataloder "chops" each track into "frames" of this size. This means that this value determines how many samples are put into the network
    overlap = 0                     # Number of overlapping samples 
                                        # Determines if the dataloader should overlap the "frames"
    data_win_type = 'boxcar'        # Window type applied to samples 
                                        # Determines if the dataloader should apply a windows to each frame. Use boxcar (Rectangular) if no window is needed
    norm_train_data = True          # Normalise samples
                                        # If true, makes sure that the L2-norm of each "frame" is 1
    batch_size = 16                 # Batch size
    shuffle = True                  # Shuffle Batches

    ###############################
    ### Optimization Parameters ###
    ###############################
    n_epochs = 20                   # Number of epochs
    learning_rate = 1e-7            # Learning rate
    beta  = 5e3                     # The weight of the MSE.
                                        # The higher the value, the higher the MSE is weighted when calculating loss
    b = 8                           # Bit depth
                                        # 2^b Discrete values produced by the quantiser
    q_nodes = 2**8                  # Number of neurons in quantization layer
                                        # Defines the bit-rate together with the bit-depth. We are sending q_nodes*b bits
    q_interval = (-1,1)             # Quantization interval/range

    prev_state = ''                 # Path to previous model parameters
                                        # NOTE that the model must fit the weight, i.e. be the same as what generated the weights

    ########################
    ### Model Parameters ###
    ########################
    
    # Defines the number of convolution blocks to use, as well as the number of kernels/channels to use for each block
    conv_features = (
        input_size//4,
        input_size,
        input_size*4
    )
    time_win_type = 'hann'          # Window applied to the MSE
                                        # When calculating the loss, a window is applied to the "frame" before calculating the MSE. To deter high frequency noise, this should weight the edge samples higher. NOTE that this is manually inverted later in the code
    kernel_size = 11                # Kernel size

    ############################
    ### Dependent Parameters ###
    ############################

    # If a Nvidia GPU is detected, use this instead of the CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device}!')

    # Finds the paths to all the MP3-files and crops the list
    paths = utils.findPathsMP3(path)
    if shuffle_tracks:
        random.seed('plosive voltages') # Hmm...
        random.shuffle(paths)
    paths = paths[:n_files]

    # Generates the needed windows.
    data_win = windows.get_window(data_win_type, input_size)

    time_win = windows.get_window(time_win_type, input_size, fftbins=False)
    time_win = torch.from_numpy(0.005/(time_win + 0.005) + 0.5).to(device)

    freq_win = torch.from_numpy(getFreqWin()).to(device)

    # Dataset and Dataloader
    dataset = data.fmaSmallDataset(paths, data_win, overlap=overlap, normalise=norm_train_data, duration=duration, offset=offset)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    # Loss Function
    loss_func = loss.MusicCompLoss(beta, time_win, freq_win)
    
    # Define the name of the weight file save after each epoch
    desc = f'Nodes_{q_nodes}__Depth_{b}'
    print(f'Now training model with q_nodes={q_nodes} and b={b}')
    
    # Model
    model = net.NeuralNetConv(input_size, b, q_interval, q_nodes, kernel_size, conv_features)

    # Loads the weights of a previous training
    if prev_state:
        model.load_state_dict(torch.load(prev_state))
        model.eval()

    # Do the training
    trainingLoop(model, train_loader, loss_func, learning_rate, n_epochs, device=device, desc=desc)