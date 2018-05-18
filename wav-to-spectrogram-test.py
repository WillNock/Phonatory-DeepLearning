"""
Author: Will Nock, ISIS, Vanderbilt School of Engineering
Speech Disorder Diagnosing with Deep Learning

File: wav-to-spectrogram-test.py :
This file is used to test pyplot/scipy libraries for processing .wav files
into spectrogram images...

These .wav files have the following properties
  Sample Rate: 44100 HZ   
  Type: 16-bit
  Channels: some Dual Channel (stereo), some Single Channel ()
  B/C spectrograms produced from the first and second channels are (near) identical,
  We only produce spectrograms from the first channel
"""


import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
#import os
#from os.path import isfile, join

#boxDirectory = "Audio cleaned alias"
#files = os.listdir(boxDirectory)
#from os import listdir
#onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]


fname = "Tremor10 - HU - rainbow.wav"
sample_rate, samples = wavfile.read(fname)
channels = samples.T

chunk_size = 3 * sample_rate # we want 3-second chunks of audio

#first_channel_chunks, second_channel_chunks = [ [channel[i:i + chunk_size] for i in range(0, len(channel), chunk_size)] for channel in channels]
first_channel_chunks = [samples[i:i + chunk_size] for i in range(0,len(samples), chunk_size)]

#Add noise to the last chunk of each channel to fill up the entire image
import random
def addNoise(chunk):
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

#first_channel_chunks[-1], second_channel_chunks[-1] = addNoise(first_channel_chunks[-1]),addNoise(second_channel_chunks[-1]) 
first_channel_chunks[-1] = addNoise(first_channel_chunks[-1])

# Plot the first channel in its own figure
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [sec]')

fig_dimension = 803/300 # we want 256x256 saved images, rounded 8/3 ratio
plt.figure(figsize=(fig_dimension,fig_dimension))
plt.specgram(first_channel_chunks[0], Fs=sample_rate, NFFT=2048, noverlap=1536, cmap=plt.get_cmap('gist_ncar'))
plt.ylim((0,5500)) # max frequency observed of 8k Hz, as seen on Audacity
                   # Note: the typical max for human speech is 5k
plt.xlim((0,3)) 
plt.axis('off')

plt.savefig("TestSecondChannel.png", dpi=96, pad_inches=0)
plt.show()

# Plot both channels in the same figure
#fig, (ax1, ax2) = plt.subplots(nrows=2)
#
#ax1.specgram(first_channel, Fs=sample_rate)
#ax1.axis('tight')
#
#ax2.specgram(second_channel, Fs=sample_rate)
#ax2.axis('tight')

# plt.title(fname)
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [sec]')

# fig.savefig("Test2.png")
