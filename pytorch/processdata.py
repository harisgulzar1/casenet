#!/usr/bin/env python

from __future__ import print_function, division
import scipy.io.wavfile as wavf
import numpy as np
from sys import argv
import os
import random

def tshift(data, sampling_rate, shift_max, shift_direction):
    #shift = np.random.randint(sampling_rate * shift_max)
    shift = int(sampling_rate * shift_max)
    if shift_direction == 'right':
        shift = -shift
    elif shift_direction == 'both':
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift
    augmented_data = np.roll(data, shift)
    # Set to silence for heading/tailing
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shift:] = 0
    return augmented_data


def manipulate(data, noise_factor):
    noise = np.random.normal(0, noise_factor, data.shape)
    #print(noise)
    augmented_data = data + noise
    #print(augmented_data)
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data

def pad_audio(data, fs, T):
    # Calculate target number of samples
    N_tar = int(fs * T)
    # Calculate number of zero samples to append
    shape = data.shape
    # Create the target shape    
    N_pad = N_tar - shape[0]
    print("Padding with %s seconds of silence" % str(N_pad/fs) )
    shape = (N_pad,) + shape[1:]
    # Stack only if there is something to append    
    if shape[0] > 0:                
        if len(shape) > 1:
            return np.vstack((data, np.zeros(shape, data.dtype)))
        else:
            return np.hstack((data, np.zeros(shape, data.dtype)))
    else:
        return data

x= 0
number = [0]*3000
def check_len(i, data, fs):
    shape = data.shape
    number[i] = shape[0]/fs
    global x
    if number[i] < 1:
        x = x+1


if __name__ == "__main__":
    cmds = ["/bed/","/bird/","/cat/","/dog/","/down/","/go/","/happy/","/house/","/left/","/marvin/","/no/","/off/","/on/","/right/","/sheila/","/stop/","/tree/","/up/","/wow/","/yes/","/zero/","/one/","/two/","/three/","/four/","/five/","/six/","/seven/","/eight/","/nine/"]
    os.chdir('/home/elsa/nakadailab/Lenetscs/speechcommands') 
    for c in range(30):
        x = 0
        aud_dir = cmds[c]
        file_names = [f for f in os.listdir("./train/" + aud_dir[1:]) if '.wav' in f]
        #file_names = [f for f in os.listdir("./train/" + aud_dir[1:]) if '.wav' in f and '_shifted' not in f]
        #file_names = [f for f in os.listdir("./") if '.wav' in f]
        i = 0
        T = 1
        print(file_names)
        """ 
        for wav_f in file_names:
            os.remove("./original" + aud_dir[1:]+wav_f)

        # Dividing Test and Train

        random.shuffle(file_names)
        nfiles = (len(file_names))
        test_list = []
        train_list = []
        test_list = file_names[0:nfiles//5]
        train_list = [x for x in file_names if x not in test_list]        

        for wfile in test_list:
            wav_f1 = os.getcwd() + "/original/" + aud_dir+ wfile
            print(wav_f1)
            fs, in_data = wavf.read(wav_f1)
            wavf.write(os.getcwd() + "/test/" + aud_dir[1:] + wfile, fs, in_data)

        for wfile in train_list:
            wav_f1 = os.getcwd() + "/original/" + aud_dir+ wfile
            print(wav_f1)
            fs, in_data = wavf.read(wav_f1)
            wavf.write(os.getcwd() + "/train/" + aud_dir[1:] + wfile, fs, in_data)

        """ 
        for wav_f in file_names:

            # Read the wav file
            fs, in_data = wavf.read(os.getcwd() + "/train" + aud_dir + wav_f)
            #fs, in_data = wavf.read("./" + wav_f)

            # Prepend with zeros
            out_data = pad_audio(in_data, fs, T)

            # Time Shift
            #out_data = tshift(in_data, fs, 0.2, 'right')

            # Adding Noise
            # out_data = manipulate(in_data, 150)
            # Save the output file
            # check_len(i, in_data, fs)
            #nfile = os.getcwd() + "/train" + aud_dir + wav_f[0:-4] + '_shifted100_'+ wav_f[-4:]
            nfile = os.getcwd() + "/train" + aud_dir + wav_f
            wavf.write(nfile, fs, out_data)
            i = i+1
            #wav_f = wav_f + "e"
        print(x)
