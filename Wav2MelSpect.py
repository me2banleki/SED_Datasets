# Recommend the user to use Ecllipse for proper sized spectrogram.
import os

import matplotlib
matplotlib.use('agg')

from matplotlib import cm
from tqdm import tqdm
import pylab

import librosa
import numpy as np

WAV_DIR = '../Wav_file_directory/'
IMG_DIR = '../Output_image_directory/'


wav_files = os.listdir(WAV_DIR)

sample_rate = 8000
window_size = 512
hop_size = 200
mel_bins = 32   #More good than 32
fmin = 5       # Hz
fmax = 4660    # Hz

for f in tqdm(wav_files):
    try:
        # Read wav-file
        y, sr = librosa.load(WAV_DIR+f, sr = sample_rate) # Use the default sampling rate of 22,050 Hz
        
        # Pre-emphasis filter
        pre_emphasis = 0.97
        y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
        
        # Compute spectrogram
        M = librosa.feature.melspectrogram(y, sr, 
                                           fmax = fmax, # Maximum frequency to be used on the on the MEL scale
                                           fmin = fmin,
                                           n_fft=window_size, 
                                           hop_length=hop_size, 
                                           n_mels = mel_bins, # As per the Google Large-scale audio CNN paper
                                           power = 2) # Power = 2 refers to squared amplitude
        
        # Plotting the spectrogram and save in .png format
        pylab.figure(figsize=(10,8))
        pylab.axis('off') 
        pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
        librosa.display.specshow(M, cmap=cm.jet)
        pylab.savefig(IMG_DIR + f[:-4]+'.png', bbox_inches=None, pad_inches=0)
        pylab.close()

    except Exception as e:
        print(f, e)
        pass
