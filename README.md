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




