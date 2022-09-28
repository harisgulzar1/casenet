from __future__ import print_function
from sys import byteorder
from array import array
from struct import pack
#from thop import profile
import torch.quantization.quantize_fx as quantize_fx
import copy
#import pyaudio
#import wave
import csv
#import matplotlib.pyplot as plt
#from scipy import signal
from scipy.io import wavfile
import os
import pandas as pd
import numpy as np
import random
#from skimage.measure import block_reduce
from numpy import newaxis

#To find the duration of wave file in seconds
import wave
import contextlib
import librosa
#import librosa.display
#from sklearn.model_selection import GridSearchCV

#Keras imports
#import keras
#from keras.models import Sequential
#from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
#from keras.models import model_from_json

import time
import datetime
import random


import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

#from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter()

device = torch.device("cuda")  # to Tensor or Module
print(device)
nfile                               = 0
os.environ['KMP_DUPLICATE_LIB_OK']  = 'True'
imwidth                             = 40
imheight                            = 40
total_examples                      = 3000
speakers                            = 6 
examples_per_speaker                = 50
tt_split                            = 0.1
num_classes                         = 10
test_rec_folder                     = "./testrecs"
log_image_folder                    = "./logims"
num_test_files                      = 1

THRESHOLD                           = 1000
CHUNK_SIZE                          = 512
#FORMAT                             = pyaudio.paInt16
RATE                                = 16000#44100
WINDOW_SIZE                         = 50
CHECK_THRESH                        = 3
SLEEP_TIME                          = 0.5 #(seconds)
IS_PLOT                             = 1
LOG_MODE                            = 0 # 1 for time, 2 for frequency


#####################################
## FUNCTIONS FOR AUDIO PROCESSING  ##
#####################################


#Check for silence
def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD

"""
Record a word or words from the microphone and 
return the data as an array of signed shorts.
"""
def record():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > 20:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    return sample_width, r

#Extract relevant signal from the captured audio
def get_bounds(ds):
    np.array(ds)
    lds = len(ds)
    count = 0
    ll=-1
    ul=-1

    #Lower Limit
    for i in range(0,lds,WINDOW_SIZE):
        sum = 0
        for k in range(i,(i+WINDOW_SIZE)%lds):
            sum = sum + np.absolute(ds[k])
        if(sum>THRESHOLD):
            count +=1
        if(count>CHECK_THRESH):
            ll = i - WINDOW_SIZE * CHECK_THRESH
            break
        
    #Upper Limit
    count = 0
    for j in range(i,lds,WINDOW_SIZE):
        sum = 0
        for k in range(j,(j+WINDOW_SIZE)%lds):
            sum = sum + np.absolute(ds[k])
        if(sum<THRESHOLD):
            count +=1
        if(count>CHECK_THRESH):
            ul = j - WINDOW_SIZE * CHECK_THRESH


        if(ul>0 and ll >0):
            break
    return ll, ul 


# Records from the microphone and outputs the resulting data to 'path'
def record_to_file(path):
    
    sample_width, data = record()
    ll, ul = get_bounds(data)
    print(ll,ul)
    if(ul-ll<100):
        return 0
    #nonz  = np.nonzero(data)
    ds = data[ll:ul]
    if(IS_PLOT):
        plt.plot(data)
        plt.axvline(x=ll)
        #plt.axvline(x=ll+5000)
        plt.axvline(x=ul)
        plt.show()

    #data = pack('<' + ('h'*len(data)), *data)
    fname = "0.wav"
    if not os.path.exists(path):
        os.makedirs(path)
    wf = wave.open(os.path.join(path,fname), 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(ds)
    wf.close()
    return 1

# Function to find the duration of the wave file in seconds
def findDuration(fname):
    with contextlib.closing(wave.open(fname,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        sw   = f.getsampwidth()
        chan = f.getnchannels()
        duration = frames / float(rate)
        #print("File:", fname, "--->",frames, rate, sw, chan)
        return duration

#Plot Spectrogram
def graph_spectrogram(wav_file, nfft=512, noverlap=511):
    findDuration(wav_file)
    rate, data = wavfile.read(wav_file)
    samples, samr = librosa.load(wav_file,sr=16000)
    #print(samples.shape[0], samr, rate, data.shape)
    vec1 = librosa.feature.mfcc(y=samples, sr=samr, hop_length = 410, n_fft=512, window='hann', n_mfcc=13)
    #print(vec1.shape)
    vec1 = vec1#subtmean(vec1)
    
    vec2 = librosa.feature.delta(vec1, order=1)

    vec2 = normalize(vec2)

    vec3 = librosa.feature.delta(vec1, width=3, order=2)
    #vec3 = normalize(vec3)

    vec4 = np.zeros(shape=(1,40))
    a = librosa.feature.rms(y=samples, frame_length=512, hop_length=410)

    for i in range(0, 37):
        vec4[:,i] = a[:, (i+1)] - a[:, i]
    #vec4 = normalize(vec4)    

    vec = np.concatenate((vec1,vec2,vec3, vec4))
    #print(vec.shape)
    
    
    # Librosa Plot
    """
    plt.figure(figsize=(40,40))
    librosa.display.specshow(vec,x_axis= "time", y_axis = "mel", sr=RATE)
    plt.colorbar(format="%+2.f")
    plt.show()
    """
       
    return vec


#Convert color image to grayscale
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

#Normalize Gray colored image
def subtmean(array):
    return (array - np.average(array))

#Normalize Gray colored image
def normalize(array):
    return (array - array.min())/(array.max() - array.min()+0.001)


#####################################
## FUNCTIONS FOR TRAINING NETWORK  ##
#####################################


#Split the dataset into test and train sets randomly
def create_train(nfile, audio_dir, label):
    file_names = [f for f in os.listdir(audio_dir) if '.wav' in f]
    # print(file_names)
    random.shuffle(file_names)
    #file_names=file_names[1:100]
    nfiles = (len(file_names))
    print(nfiles)

    y_train = np.zeros(len(file_names))
    x_train = np.zeros((len(file_names), imheight, imwidth))


    for i, f in enumerate(file_names):
        y_train[i] = label
        spectrogram   = graph_spectrogram( audio_dir + f )
        x_train[i,:,:] = spectrogram

        print("Progress Training Data: {:2.1%}".format(float(i) / len(file_names)), end="\r")
    return x_train, y_train


def create_test(nfile, audio_dir, label):
    file_names = [f for f in os.listdir(audio_dir) if '.wav' in f]
    # print(file_names)
    random.shuffle(file_names)
    #file_names=file_names[1:100]
    nfiles = (len(file_names))
    print(nfiles)

    y_test = np.zeros(len(file_names))
    x_test = np.zeros((len(file_names), imheight, imwidth))


    for i, f in enumerate(file_names):
        y_test[i]     = label
        spectrogram   = graph_spectrogram( audio_dir + f )

        data          = np.empty((1,1,imheight,imwidth))
        data[0,:,:,:] = spectrogram
        np.savetxt('./testdata40p16/digit'+str(nfile)+'_label_'+str(label)+'.txt', np.ravel(np.array(data[0,0,:,:])),fmt='%.16f')
        nfile = nfile + 1
        x_test[i,:,:] = spectrogram
        print("Progress Test Data: {:2.1%}".format(float(i) / len(file_names)), end="\r")

    return x_test, y_test

#Create Keras Model
def create_model(path):
    lab =1
    x_train, y_train, x_test, y_test = create_train_test(path,lab)
    
    

    print("Size of Training Data:", np.shape(x_train[0]))
    print("Size of Training Labels:", np.shape(y_train[0]))
    print("Size of Test Data:", np.shape(x_test[0]))
    print("Size of Test Labels:", np.shape(y_test[0]))

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    x_train = x_train.reshape(x_train.shape[0], imheight, imwidth, 1)
    x_test = x_test.reshape(x_test.shape[0], imheight, imwidth, 1)
    input_shape = (imheight, imwidth, 1)
    batch_size = 4
    epochs = 1

    model = Sequential()
    model.add(Conv2D(20, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(50, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(800, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    print(model.summary())

    model.fit(x_train, y_train, batch_size=4, epochs=10, verbose=1, validation_data=(x_test, y_test))
    return model

#Extract wave data from recorded audio
def get_wav_data(path):
    input_wav           = path
    spectrogram         = graph_spectrogram( input_wav )
    """
    print("RGB Dim: ", spectrogram.shape)
    graygram            = rgb2gray(spectrogram)
    print("Gray Scale Dim: ", graygram.shape)
    normgram            = normalize_gray(graygram)
    norm_shape          = normgram.shape
    print("Norm. Gray: ", norm_shape)
    redgram             = block_reduce(normgram, block_size = (17,22), func = np.mean)
    print("Reduced Gray: ", redgram.shape)
    redgram             = redgram[0:imheight,0:imwidth]
    red_data            = redgram.reshape(1,imheight,imwidth)
    """
    empty_data          = np.empty((1,1,imheight,imwidth))
    empty_data[0,:,:,:] = spectrogram
    new_data            = empty_data
    
    return new_data

#Save created model
def save_model_to_disk(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

#Load saved model
def load_model_from_disk():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return loaded_model

#In Loggin mode capture one example of each class. Display it in time and frequency domain.
def generate_log(in_dir, num_samps_per_cat):
    file_names = [f for f in os.listdir(in_dir) if '.wav' in f]
    checklist  = np.zeros(num_samps_per_cat * 10)
    final_list = []
    iternum = 0
    
    #Get a random sample for each category
    while(1):
        print("Iteration Number:", iternum)
        sample_names = random.sample(file_names,10)
        for name in sample_names:
            categ = int(name[0])
            if(checklist[categ]<num_samps_per_cat):
                checklist[categ]+=1
                final_list.append(name)
        if(int(checklist.sum())==(num_samps_per_cat * 10)):
            break 
        iternum+=1
    print(final_list)

    #Generate Images for each sample
    lif = os.path.join(log_image_folder,time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime()))
    if not os.path.exists(lif):
        os.makedirs(lif)
    for name in final_list:      
        #Time Domain Signal
        rate, data = wavfile.read(os.path.join(in_dir,name))
        if(LOG_MODE==1):   
            fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
            ax.set_title('Sound of ' +name[0] + ' - Sampled audio signal in time')
            ax.set_xlabel('Sample number')
            ax.set_ylabel('Amplitude')
            ax.plot(data)
            fig.savefig(os.path.join(lif, name[0:5]+'.png'))   # save the figure to file
            plt.close(fig)
    
        #Frequency Domain Signals
        if(LOG_MODE==2):
            fig,ax = plt.subplots(1)
            #fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
            #ax.axis('off')
            pxx, freqs, bins, im = ax.specgram(x=data, Fs=rate, noverlap=511, NFFT=512)
            #ax.axis('off')
            #plt.rcParams['figure.figsize'] = [0.75,0.5]
            cbar = fig.colorbar(im)
            cbar.set_label('Intensity dB')
            #ax.axis("tight")

            # Prettify
            ax.set_title('Spectrogram of spoken ' +name[0] )
            ax.set_xlabel('time')
            ax.set_ylabel('frequency Hz')
            fig.savefig(os.path.join(lif, name[0]+'_spec.png'), dpi=300, frameon='false')
            plt.close(fig)

######################################################
## DATA LOADER FUNCTION FOR GOOGLE SPEECH COMMANDS  ##
######################################################


class SpokenDigits(torch.utils.data.Dataset):

    def __init__(self, x_vect, y_vect):
      x_tmp = x_vect
      y_tmp = y_vect

      self.x_data = torch.tensor(x_tmp, dtype=torch.float32).to(device)
      self.y_data = torch.tensor(y_tmp, dtype=torch.long).to(device)

    def __len__(self):
      return len(self.x_data)  # required

    def __getitem__(self, idx):
      if torch.is_tensor(idx):
        idx = idx.tolist()
      sp_digit = self.x_data[idx,newaxis,:,:]
      label = self.y_data[idx]
      sample = \
        { 'spokendigit' : sp_digit, 'label' : label }
      return sample


#####################################
## PYTORCH MODELS DEFINITION       ##
#####################################



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1,padding=(0, 0))
        self.conv2 = nn.Conv2d(32,16, 5, 1,padding=(0, 0))
        #self.conv3 = nn.Conv2d(20,50, 3, 1)
        #self.conv3 = nn.Conv2d(32, 16, 3, 1)
        #self.dropout1 = nn.Dropout(0.25)
        #self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(784,100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.conv1(x)
        #print(x.shape)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        #x = self.dropout1(x)
        #print(x.shape)
        x = self.conv2(x)
        #print(x.shape)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        #print(x.shape)
        #x = self.dropout1(x)
        #print(x.shape)
        """
        x = self.conv3(x)
        #print(x.shape)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        #print(x.shape)
        x = self.dropout1(x)
        #print(x.shape)
        """
        x = torch.flatten(x, 1)
        #print(x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        #x = self.dropout2(x)
        #print(x.shape)
        x = self.fc2(x)
        #print(x.shape)
        output = F.log_softmax(x, dim=1)
        return output

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.bnorm1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32,64, 3, 1)
        self.bnorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,32, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.bnorm3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(288,100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.bnorm1(x)
        x = self.conv1(x)
        #print(x.shape)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        #print(x.shape)
        x = self.conv2(x)
        #print(x.shape)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        #print(x.shape)
        x = self.dropout1(x)
        #print(x.shape)

        x = self.bnorm2(x)
        x = self.conv3(x)
        #print(x.shape)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        #print(x.shape)
        x = self.dropout1(x)
        #print(x.shape)
        x = self.bnorm3(x)
        x = torch.flatten(x, 1)
        #print(x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        #print(x.shape)
        x = self.fc2(x)
        #print(x.shape)
        output = F.log_softmax(x, dim=1)
        return output

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.bnorm1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 128, 5, 1)
        self.conv2 = nn.Conv2d(128,128, 3, 1)
        self.bnorm2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128,128, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.bnorm3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(1152,256)
        self.fc2 = nn.Linear(256, 7)

    def forward(self, x):
        x = self.bnorm1(x)
        x = self.conv1(x)
        #print(x.shape)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        #print(x.shape)
        x = self.conv2(x)
        #print(x.shape)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        #print(x.shape)
        x = self.dropout1(x)
        #print(x.shape)

        x = self.bnorm2(x)
        x = self.conv3(x)
        #print(x.shape)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        #print(x.shape)
        x = self.dropout1(x)
        #print(x.shape)
        x = self.bnorm3(x)
        x = torch.flatten(x, 1)
        #print(x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        #print(x.shape)
        x = self.fc2(x)
        #print(x.shape)
        output = F.log_softmax(x, dim=1)
        return output

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        data = batch['spokendigit']
        target = batch['label']
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        #writer.add_scalar("Loss/train", loss, batch_idx)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(i, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in enumerate(test_loader):
            data = target['spokendigit']
            target = target['label']
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            #print(output)
            #pred_np = output.cpu().numpy()
            #pred_df = pd.DataFrame(pred_np)
            #pred_df.to_csv('predictionsnew.csv', mode='a', header=False)
    test_loss /= len(test_loader.dataset)
    #f.close()
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    #writer.add_scalar("Test Loss", test_loss, i)
    #writer.add_scalar("Accuracy", 100. * correct / len(test_loader.dataset), i)
        


#####################################
## MAIN FUNCTION FOR EXECUTION     ##
#####################################


if __name__ == '__main__':

# ---------------------------------------------------

    

    

    print("\n==============================")
    
    
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                         help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=1.0, metavar='M',
                        help='Learning rate step gamma (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 0,
                       'pin_memory': False,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
        print("GPU")

    """transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    """
    
      
    print("\nBegin PyTorch Demo ")




    cmds = ["/yes/","/no/","/up/","/down/","/left/","/right/","/on/","/off/","/stop/","/go/"]#,"/zero/","/one/","/two/","/three/","/four/","/five/","/six/","/seven/","/eight/","/nine/"]
    #cmds = ["/up/","/down/","/left/","/right/","/stop/","/go/","/no/", "/off/", "/on/","/yes/","/zero/","/one/","/two/","/three/","/four/","/five/","/six/","/seven/","/eight/","/nine/"]
    train_directory = "./speechcommands/train"
    test_directory  = "./speechcommands/test"

    ## Code to process the speech commands dataset and save in .pt file

    """
    # 1. create Dataset and DataLoader object
    print("\nCreating Dataset and DataLoader ")
    label = 0
    x_train, y_train = create_train(nfile, train_directory+cmds[0], 0)
    x_test, y_test = create_test(nfile, test_directory+cmds[0], 0)


    for c in range(9):
        label = c+1
        x_train1, y_train1 = create_train(nfile, train_directory+cmds[c+1],label)
        x_test1, y_test1 = create_test(nfile, test_directory+cmds[c+1],label)

        x_train = np.concatenate((x_train,x_train1))
        y_train = np.concatenate((y_train,y_train1))
        x_test = np.concatenate((x_test,x_test1))
        y_test = np.concatenate((y_test,y_test1))
        print(x_train.shape, x_test.shape, label)
    

    train_ds = SpokenDigits(x_train, y_train)
    test_ds = SpokenDigits(x_test, y_test)
    
    torch.save(train_ds, "train.pt")
    torch.save(test_ds, "test.pt")
    """


    train_ds = torch.load("trainadamold.pt")
    test_ds = torch.load("testadamold.pt")
    
    
    train_loader = torch.utils.data.DataLoader(train_ds, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_ds, **test_kwargs)
    
    model = Net1().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00045, amsgrad=False)#Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

   
    #writer.flush()
    #writer.close()

    model.load_state_dict(torch.load("Net1.pt"))
    model.to(device)
    model.eval()

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(epoch, model, device, test_loader)
        #scheduler.step()
    
    if args.save_model:
        torch.save(model.state_dict(), "Net1.pt")

    # Printing and saving weights of network

    params_dir = 'weights/'
    for key in model.state_dict():
        #print(np.ravel(np.array(model.state_dict()[key])))
        model.cpu()
        np.savetxt(params_dir + key+ '.txt', np.ravel(np.array(model.state_dict()[key])),fmt='%.8f')
    #for var_name in optimizer.state_dict():
        #print(var_name, "\t", optimizer.state_dict()[var_name])

recording_directory = './speechcommands/'    


for i in range(10):
    file_names = [f for f in os.listdir(test_directory+cmds[i]) if '.wav' in f]
    # print(file_names)
    random.shuffle(file_names)
    file_names=file_names[1]
    print("File: ", test_directory + cmds[i]+file_names)
    new_data = get_wav_data(test_directory + cmds[i] + file_names)
    #print(new_data)

    ###############################################################
    # SAVE THE AUDIO SIGNAL AS TEXT FILE AFTER TAKING SPECTROGRAM #
    ###############################################################
 
    np.savetxt('./test_mk/digit00'+str(i)+'.txt', np.ravel(np.array(new_data[0,0,:,:])),fmt='%.8f')


    #print(np.shape(new_data))
    model.double()
    pred_img = torch.tensor(new_data)
    pred_img = pred_img.type(torch.double)
    pred_img.to('cpu')
    model.to('cpu')

    predictions = model(pred_img)
    #flops, params = profile(model, inputs=(pred_img, ))
    #print (predictions)
    pred = predictions.max(1, keepdim=True)[1]
    print("Prediction: ", pred)
    print("_____________________________\n")
    #print(flops, params)



"""
tuned_parameters = [
    {'lr': [0.0005, 0.0002, 0.0001]}
    ]
    
model = Net1().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)#Adadelta(model.parameters(), lr=args.lr)


clf = GridSearchCV(
    optimizer, 
    tuned_parameters,
    cv=5, 
    scoring='f1' )

clf.fit(train_ds)


num = len(clf.cv_results_['params'])
for i in range(num):
    print(i,"\t",clf.cv_results_['params'][i],"\t",clf.cv_results_['mean_test_score'][i])
        

print("Best Parameter:",clf.best_params_)
"""
