<img align="right" src="assets\images\aau_logo.png">

# Information Bottleneck Principle Applied to PyTorch Autoencoder 

The main idea is to use the amplitude spectrum to generate the probability distributions used for the cross entropy. Combine this with the MSE and we have a *(unique)* loss function. 

## Usage
For training the [FMA dataset](https://github.com/mdeff/fma) is used. To load this, a custom dataloader based on [Librosa](https://librosa.org/doc/main/index.html) is utilised. To recreate the Python environment, use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (or [Anaconda](https://www.anaconda.com/)) with the *requirements.txt* file.

*main.py* contains the code to run the script. It is designed to be run from an editor, such as [Spyder](https://www.spyder-ide.org/) or [Visual Studio Code](https://code.visualstudio.com/), in order to change the parameters. The comments in the script should explain what the different parameters does and when to change them.

## About
This code was written as part of a 7th semester project in *Mathematical Engineering* as Aalborg University, Denmark.

Code and associated paper is composed by Alexander F, Andreas L, Gustav Z, Mads J & Magnus L.


