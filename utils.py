import os
import sys
import warnings
import librosa

import numpy as np

from itertools import repeat
from p_tqdm import p_umap

def findPathsMP3(folder, filetype='.mp3'):
    """
    Finds all MP3s in the folder and subsequent subfolders.

    Args:
        folder (str): Top folder to start searching for MP3 files.

    Returns:
        paths (list): Relative paths to all the found MP3 files.
    """

    tree = [i for i in os.walk(folder)]
    paths = []
    for branch in tree:
        
        parents = branch[0]
        files = branch[2]
        
        for f in files:
            if f[-4:] == filetype:
                path = parents + '/' + f
                paths.append(path)

    return paths


def loadMP3(path, duration=None, offset=0.0):
    """
    Loads a single MP3 file from a given path. Uses duration and offset to determine what part of the signal to load.

    Args:
        path (str): Path to the MP3 file.
        duration (float): Only load up to this much audio (seconds).
        offset (float): Start reading after this time (seconds).

    Returns:
        mp3 (np.array): Array (range: [-1,1]) representing the waveform. Always Mono.
    """

    # Supress warnings a "pysoundfile" does not have MP3 support yet
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            mp3, _ = librosa.load(path, sr=None, mono=True, duration=duration, offset=offset)
        except:
            print(f'File {path} could not be loaded')
    return mp3


def loadAllMP3s(paths, duration=None, offset=0.0):
    """
    Loads all the MP3 files from the given paths. Uses duration and offset to determine what part of the signal to load.

    Args:
        paths (list): Paths to the MP3 files 
        duration (float, optional): Only load up to this much audio (seconds). Defaults to None.
        offset (float, optional): Start reading after this time (seconds). Defaults to 0.0.

    Returns:
        [type]: [description]
    """

    print('Loading Files...')
    if sys.platform == 'darwin':
        mp3s = [loadMP3(i, duration, repeat) for i in paths]
    else:
        mp3s = p_umap(loadMP3, paths, repeat(duration), repeat(offset))

    return mp3s


if __name__ == '__main__':

    paths = findPathsMP3('fma_small')[:100]
    data = loadAllMP3s(paths, duration=1, offset=10)