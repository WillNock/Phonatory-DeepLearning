"""
Author: Will Nock, ISIS, Vanderbilt School of Engineering
Speech Disorder Diagnosing with Deep Learning

File: produce_spectrograms.py :
This file produces spectrograms from .wav files in the project BOX Folder.

"""


from os import listdir, mkdir
from os.path import join

working_dir_path = "/Users/Will/ISIS/Speech Therapy - Deep Learning/"
box_folder_path = "/Users/Will/Box Sync/"
audio_dirs = [d for d in listdir(box_folder_path)]

# Remove the DS_Store directory...
audio_dirs.remove(".DS_Store")
# Exclude iphone spectrograms for now...
audio_dirs.remove("iphone nstream comparison")


#Adds noise to the last chunk of each channel (*to fill up the entire image)
import random
def addNoise(chunk, chunk_size):
    """ 
    This function adds noise to the last 'chunk' of a
    signal channel in order to preserve a uniform length (time) of each chunk of sound.
    Noise is produced by repeatedly appending a copy signal of +-5% error for each signal frequency.
    """
    chunk = chunk.tolist()
    raw_size = len(chunk)
    counter = raw_size
    while (counter < chunk_size):
        for i in range(raw_size):
            if (counter == chunk_size):
                break
            chunk.append(int(chunk[i]*(random.uniform(.95,1.05))))
            counter += 1
        if (counter == chunk_size):
            break
    return np.array(chunk)


import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np

fig_dimension = 803/300 # we want 256x256 saved images, rounded 8/3 ratio

for directory in audio_dirs:
    
    # Make proper spectrogram directories if necessary
    spec_dir_path = join(box_folder_path, directory) + "_spec"
    if directory+"_spec" not in listdir(box_folder_path):
        mkdir(spec_dir_path)  
    
    for audio_file in listdir(join(box_folder_path, directory)):
        specs = audio_file.replace(".wav","_spec")
        if specs not in listdir(spec_dir_path):
            mkdir(join(spec_dir_path, specs))
        
        full_audio_path = join(box_folder_path,join(directory,audio_file))
        sample_rate, samples = wavfile.read(full_audio_path)
        
        if len(samples.T) == 2:
            samples = samples.T[0]
        
        # We want spectrograms for each 3-second chunk of audio
        chunk_size = 3 * sample_rate  
        chunks = [samples[i:i + chunk_size] for i in range(0,len(samples), chunk_size)]
        # Add noise to the last chunk
        chunks[-1] = addNoise(chunks[-1], chunk_size)
        
        # Produce a Spectrogram for each chunk of audio
        for i in range(len(chunks)):
            plt.figure(figsize=(fig_dimension,fig_dimension))
            plt.specgram(chunks[i], Fs=sample_rate, NFFT=2048, noverlap=1536, cmap=plt.get_cmap('gist_ncar'))
            plt.ylim((0,5500)) # max frequency observed of 8k Hz, as seen on Audacity
                               # Note: the typical max for human speech is 5k...
            plt.xlim((0,3))    # 3-second chunks
            plt.axis('off')
            full_spec_path = join(spec_dir_path,specs) + "/" + specs + "_" + str(i+1) + ".png"
            plt.savefig(full_spec_path, dpi=96, pad_inches=0)
            plt.close()