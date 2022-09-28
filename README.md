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
