# CASENet
## CASE: CNN Acceleration for Speech-Classification in Edge-Computing ([Paper](https://ieeexplore.ieee.org/document/9658881))

This repository is the guidance for the implementation of casenet on [M-KUBOS](https://www.paltek.co.jp/design/original/m-kubos/index.html) Device.

There are three step to implement Machine Learning (ML) model on M-KUBOS Device:

1. Training the CNN model in PyTorch framework.
2. Making the Hardware logic for M-KUBOS FPGA using Vivado.
3. Implementating the Python application program on the M-KUBOS Device.

### Part 1: CASENet Pytorch

CASENet model is trained on [Google Speech Commands Dataset](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html).
This dataset contains several spoken speech commands, but this work only uses only 10 speech commands from those.
This dataset has 1 sec long audio file for each spoken commands.

General flow is as follows:
1. Install necessary packages in your machine i.e. pytorch, librosa etc.
2. Download the data from this GD [link](https://drive.google.com/drive/folders/15kztVyflMU2n-_H_jRNFERxEsEJTfBAB?usp=sharing). This is already pre-processed a little bit to make the lengths of all audio files uniform.
3. Data augmentation can be applied to the audio file i.e. time shifting, noise addition etc. using [processdata.py](https://github.com/harisgulzar1/casenet/blob/main/pytorch/processdata.py) file.
4. The CNN model and all relevant function for training the model on Google Speech Commands are defined in main file [main_wd.py](https://github.com/harisgulzar1/casenet/blob/main/pytorch/main_wd.py) file.
5. This file will save the model weights in text form also you can save the audio files to test on M-KUBOS using this file.

### Part 2: HLS Logic Design for FPGA part of M-KUBOS

1. [HLS](https://github.com/harisgulzar1/casenet/tree/main/HLS) contains the code HLS logic.
(This code is written for CNN network with 3 convolutional layers. The structure of the network can be changed in C code but it should be same as the network which was trained using Pytorch in part 1.)
2. Next part is to use the Vivado software to generate hardware file for FPGA.
(Instructions to use Vivado software are explained in [power point tutorial](https://drive.google.com/file/d/12tVDj-0U91x_nHXJfFPUi4-i0-DsW-Oo/view?usp=sharing).)

### Part 3: Application program on the device

By the point you have:
1. Weights of the CNN network in .txt file.
2. Input files for testing the network on M-KUBOS in .txt file.
3. Hardware files for the FPGA (.bit and .hwh files).

The next step is to transfer these files to M-KUBOS and run the applications program [CASENet2.py](https://github.com/harisgulzar1/casenet/blob/main/mukobosprogram/CASENet2.py)

This file uses pynq library to use on-chip FPGA and send/receive data from ARM to FPGA part.
There are many variables in the application program according to the size of CNN, these parameters should exactly be same as CNN parameters in HLS file and in Pytorch model file.
