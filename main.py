import torch
import random

import numpy as np

from tqdm import tqdm
from scipy.signal import windows
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Custom packages
import net
import data
import utils
import loss


def getFreqWin():
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

    # Prints the model parameters
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    writer = SummaryWriter()

    for epoch in range(n_epochs):
        
        loss_total = 0
        mse_total = 0
        ce_total = 0

        pbar = tqdm(total=len(dataloader), dynamic_ncols=True)
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
            
            loss_total += loss.item()
            mse_total += mse.item()
            ce_total += ce.item()

            if (i + 1) % 50 == 0:
                writer.add_scalar(f'{desc}/{epoch + 1}/Loss', loss.item(), i)
                writer.add_scalar(f'{desc}/{epoch + 1}/Loss_MSE', mse.item(), i)
                writer.add_scalar(f'{desc}/{epoch + 1}/Loss_CE', ce.item(), i)
                pbar.set_postfix({'Avg Loss':f'{loss_total/(i+1):.8f}' , 'Avg MSE': f'{mse_total/(i+1):.8f}', 'Avg CE': f'{ce_total/(i+1):.8f}'})
                pbar.update(50)

        ############## SAVING MODEL ###################
        pbar.close()
        torch.save(model.state_dict(), f'weights_{desc}.pth')


if __name__ == '__main__':
    ##########################
    ### Dataset Parameters ###
    ##########################
    path = 'fma_small'              # Path to fma_small dataset (larger might also work)
    n_files = 2000                  # Number of files to be included in the training data
    duration =  1                   # duration of each file in seconds, set to None if whole duration of music files should be included
    offset = 10                     # from which second the sound file should load from, be careful wrt chosen duration contra offset

    input_size = 2**7               # Window length of each input in samples
    overlap = 0                     # Size of window overlap in samples
    data_win_type = 'boxcar'        # Window type (Rectangular = boxcar)

    norm_train_data = True          # Normalise training data (normalises each window in training set)
    batch_size = 16                 # Batch size of training data
    shuffle = True                  # Shuffle batches for every epoch

    shuffle_paths = True            # Shuffle which music files to import

    ###############################
    ### Optimization Parameters ###
    ###############################
    n_epochs = 20                   # Number of epochs
    learning_rate = 1e-7            # Learning rate
    beta  = 5e3                     # The weight of the noise exponential in loss func

    b = 8                           # Bit depth
    q_interval = (-1,1)             # Quantization interval
    q_nodes = 2**8                  # Number of neurons in quantization layer

    prev_state = ''                 # Path to previous model parameters. If empty the NN starts from scratch

    ########################
    ### Model Parameters ###
    ########################
    
    # NOTE: len(conv_features) = #Conv_blocks, conv_feautures[i] = #channels
    conv_features = (
        input_size//4,
        input_size,
        input_size*4
    )
    time_win_type = 'hann'
    kernel_size = 11

    ############################
    ### Dependent Parameters ###
    ############################



    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'Using {device}!')

    paths = utils.findPathsMP3(path)
    if shuffle_paths:
        random.seed('plosive voltages')
        random.shuffle(paths)
    paths = paths[:n_files]

    data_win = windows.get_window(data_win_type, input_size)

    time_win = windows.get_window(time_win_type, input_size, fftbins=False)
    time_win = torch.from_numpy(0.005/(time_win + 0.005) + 0.5).to(device)

    freq_win = torch.from_numpy(getFreqWin()).to(device)

    # Dataset and Dataloader
    dataset = data.fmaSmallDataset(paths, data_win, overlap=overlap, normalise=norm_train_data, duration=duration, offset=offset)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    # Loss Function
    loss_func = loss.MusicCompLoss(beta, time_win, freq_win)
    
    parameter_list = [[2**8,8],[2**3,16],[2**4,16],[2**5,16],[2**7,16]]
    # for pars in parameter_list[-1]:
    q_nodes, b = 2**7, 16
    desc = f'Nodes_{q_nodes}__Depth_{b}'
    print(f'Now training model with q_len={q_nodes} and b={b}')
    
    # Model
    model = net.NeuralNetConv(input_size, b, q_interval, q_nodes, kernel_size, conv_features)

    if prev_state:
        model.load_state_dict(torch.load(prev_state))
        model.eval()

    trainingLoop(model, train_loader, loss_func, learning_rate, n_epochs, device=device, desc=desc)