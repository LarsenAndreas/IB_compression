import torch

import numpy as np

from tqdm import tqdm
from utils import loadAllMP3s, findPathsMP3
from torch.utils.data import Dataset
from scipy.signal import windows


class fmaSmallDataset(Dataset):
    """
    Loads the data from the fmaSmall dataset as a Pytorch Dataset.
    """
    
    def __init__(self, paths, window, overlap=0, normalise=False, duration=None, offset=0.0):
        """
        Args:
            paths (list): List of paths to MP3 files.
            window (np.array): The window coefficient. Length corresponds to the number of samples used in each "frame" cut out from the signal.
            overlap (int, optional): Number of samples to overlap between frames. Defaults to 0.
            normalise (bool, optional): If each frame should be normalised. Defaults to False.
            duration (float, optional): Number of seconds to include from the signal. Max 30s. If "None" includes the entire signal. Defaults to None.
            offset (float, optional): Which second to start reading from in the signal. Defaults to 0.0.
        
        Note:
            duration + offset < 30
        """

        # Finds all the paths to the MP3 files
        self.files = loadAllMP3s(paths, duration=duration, offset=offset)
        
        # Progress bar stuff
        pbar = tqdm(total=len(self.files))
        pbar.set_description('Butchering')
        
        # Cuts the original signals into frames and applies normalisation and window functions
        frames = []
        for mp3 in self.files:
            
            try: #  Some files are corrupted, this will prevent the program from halting
                frames.append(self._butcher(mp3, window, overlap, normalise))
            except:
                print('Error', mp3)
            
            pbar.update()
        
        pbar.set_description('Finished Butchering')
        pbar.close()
        
        self.items = torch.from_numpy(np.vstack(frames))
        
    
    def __len__(self):
        return len(self.items)
    
    
    def __getitem__(self, index):
        return self.items[index]
    
    
    def _butcher(self, mp3, window, overlap, normalise):
        """
        Splits a single signal into frames, such that each frame contains the same number of samples as "window". The window is then multiplied onto the frame. If normalisation is True, then each frame is normalised as well.
        """
        
        # Cuts the signal into the correct sizes and stores it in a list
        win_len = len(window)
        if overlap > 0:
            step = win_len - overlap
            temp = []
            i = 0
            j = win_len
            while j <= len(mp3):
                temp += list(mp3[i:j])
                i += step
                j += step
            
            mp3 = np.array(temp, dtype=np.float32)
        
        rows = len(mp3)//win_len
        max_samples = win_len*rows
        cut_mp3 = mp3[:max_samples]
        butch = cut_mp3.reshape(rows, win_len)*window
        
        # Normalises each signal such individually
        if normalise==True:
            for i in range(len(butch)):
                norm = np.linalg.norm(butch[i], ord=2)
                if norm != 0:
                    butch[i] /= norm

        return butch.astype(np.float32)


if __name__ == '__main__':

    fma = 'fma_small'
    paths = findPathsMP3(fma)[:10]
    window = windows.get_window('boxcar', 50)
    
    dataset = fmaSmallDataset(paths, window, overlap=0, normalise=True, duration=3, offset=10)
      
    print(dataset[0])

    print('Ran!')