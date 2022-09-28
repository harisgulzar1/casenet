#! /usr/bin/python3.6
# -*- coding: utf-8 -*-
# mras test

import random
import time
from pynq import Overlay
from pynq import MMIO
#from mmio2b import MMIO2
from mcube10fpga import *
import struct
import os

IMAGE_SIZE = 1*40*40
CONV1_W_SIZE = 64*1*5*5
CONV1_B_SIZE = 64
CONV1_OUT_SIZE = 64*36*36
POOL1_OUT_SIZE = 64*18*18

CONV2_W_SIZE = 128*64*3*3
CONV2_B_SIZE = 128
CONV2_OUT_SIZE = 128*16*16

POOL2_OUT_SIZE = 128*8*8

CONV3_W_SIZE = 64*128*3*3
CONV3_B_SIZE = 64
CONV3_OUT_SIZE = 64*6*6

POOL3_OUT_SIZE = 64*3*3

FC1_W_SIZE = 128*576
FC1_B_SIZE = 128
FC1_OUT_SIZE = 128
FC2_W_SIZE = 10*128
FC2_B_SIZE = 10
FC2_OUT_SIZE = 10

RESULT_SIZE = 12

ALL_WB_SIZE = (CONV1_W_SIZE+CONV1_B_SIZE+CONV2_W_SIZE +
               CONV2_B_SIZE+CONV3_W_SIZE+CONV3_B_SIZE+FC1_W_SIZE+FC1_B_SIZE+FC2_W_SIZE+FC2_B_SIZE)

OFFSET_CTRL = 0x000000
OFFSET_INPUT = 0x002000
OFFSET_OUTPUT = 0x004000
OFFSET_WB = 0x100000


def read_params(fname, array, size):
    for line in open(fname).readlines():
        bdata = struct.pack('>f', float(line))
        idata = int.from_bytes(bdata, 'big')
        array.append(idata)


def write_params(offset, array, size):
    i = 0
    while(i < size):
        lenet_mmio.write(offset+i*4, array[i])
        i = i+1


def read_image(fname, array, size):
    for line in open(fname).readlines():
        bdata = struct.pack('>f', float(line))
        idata = int.from_bytes(bdata, 'big')
        array.append(idata)

def show_image(array, size):
    i = 0
    while(i < size):
        j = 0
        while(j < size):
            if(array[i*size+j] > 0.5):
                print("* ", end="")
            else:
                print("  ", end="")
            j = j+1
        print("")
        i = i+1

# Which IP to Execute


IPv = "IPv22_3LP"

# FPGA bit file
BIT_FILE = IPv+"/IP/lenet.bit"

print("downloading {}".format(BIT_FILE))
ovl = Overlay(BIT_FILE)
# show the IP names
print("ovl.ip_dict.keys() = ", ovl.ip_dict.keys())

# check AXI-Lite registers
regs = ovl.ip_dict["lenetcon_0"]
phys_addr = regs["phys_addr"]
addr_range = regs["addr_range"]
print("lenet:phys_addr = 0x{:X}".format(phys_addr))
print("lenet:addr_range = {}".format(addr_range))
lenet_mmio = MMIO(phys_addr, addr_range)
time_start = time.perf_counter()


print("setting conv1 weight data")
conv1_w = []
offset = OFFSET_WB
read_params("./"+IPv+"/weights/conv1.weight.txt", conv1_w, CONV1_W_SIZE)
write_params(offset, conv1_w, CONV1_W_SIZE)
offset = offset+4*CONV1_W_SIZE

conv1_b = []
read_params("./"+IPv+"/weights/conv1.bias.txt", conv1_b, CONV1_B_SIZE)
write_params(offset, conv1_b, CONV1_B_SIZE)
offset = offset+4*CONV1_B_SIZE

print("setting conv2 weight data")
conv2_w = []
read_params("./"+IPv+"/weights/conv2.weight.txt", conv2_w, CONV2_W_SIZE)
write_params(offset, conv2_w, CONV2_W_SIZE)
offset = offset+4*CONV2_W_SIZE

conv2_b = []
read_params("./"+IPv+"/weights/conv2.bias.txt", conv2_b, CONV2_B_SIZE)
write_params(offset, conv2_b, CONV2_B_SIZE)
offset = offset+4*CONV2_B_SIZE

print("setting conv3 weight data")
conv3_w = []
read_params("./"+IPv+"/weights/conv3.weight.txt", conv3_w, CONV3_W_SIZE)
write_params(offset, conv3_w, CONV3_W_SIZE)
offset = offset+4*CONV3_W_SIZE

conv3_b = []
read_params("./"+IPv+"/weights/conv3.bias.txt", conv3_b, CONV3_B_SIZE)
write_params(offset, conv3_b, CONV3_B_SIZE)
offset = offset+4*CONV3_B_SIZE

print("setting fc1 weight data")
fc1_w = []
read_params("./"+IPv+"/weights/fc1.weight.txt", fc1_w, FC1_W_SIZE)
write_params(offset, fc1_w, FC1_W_SIZE)
offset = offset+4*FC1_W_SIZE

fc1_b = []
read_params("./"+IPv+"/weights/fc1.bias.txt", fc1_b, FC1_B_SIZE)
write_params(offset, fc1_b, FC1_B_SIZE)
offset = offset+4*FC1_B_SIZE

print("setting fc2 weight data")
fc2_w = []
read_params("./"+IPv+"/weights/fc2.weight.txt", fc2_w, FC2_W_SIZE)
write_params(offset, fc2_w, FC2_W_SIZE)
offset = offset+4*FC2_W_SIZE

fc2_b = []
read_params("./"+IPv+"/weights/fc2.bias.txt", fc2_b, FC2_B_SIZE)
write_params(offset, fc2_b, FC2_B_SIZE)
offset = offset+4*FC2_B_SIZE

time_end = time.perf_counter()
time_span = time_end - time_start
print("time_span = {:.3f} [us]".format(time_span * 1e6))

file_names = [f for f in os.listdir('./testdata40p16/') if '.txt' in f]
correct = 0
nfiles  = 0
labels = ["yes","no","up","down","left","right","on","off","stop","go"]

for f in file_names:
    nfiles = nfiles+1
    t1 =  time.perf_counter() #Start Time
    label = f[-5]
    if(nfiles == 20):
        break
    image = []
    # image1='./IPv8_simplest/test_mk/image.txt'
    read_image('./testdata40p16/'+f, image, IMAGE_SIZE)
    # read_image(image3,image,IMAGE_SIZE)
    write_params(OFFSET_INPUT, image, IMAGE_SIZE)
    #print("start computation", image1)
    lenet_mmio.write(OFFSET_CTRL, 0x01)

    while 1:
        status=lenet_mmio.read(OFFSET_CTRL)
        #print("status:", status)
        if((status & 0x2) == 0x2):
            break
        #time.sleep(1)

    # print ("Possibility:")
    rr=0
    r=[]
    while rr < RESULT_SIZE-2:
        itmpl=lenet_mmio.read(OFFSET_OUTPUT+rr*4)
        bdata=struct.pack('>I', itmpl)
        result=struct.unpack('>f', bdata)
        # print (rr,result[0])
        r.append(result[0])
        rr=rr+1
    prob = max(r)
    plabel = r.index(prob)
    #print(plabel)
    t2 =  time.perf_counter() #End Time
    latency = t2-t1
    print("Latency = {:.3f} [us]".format(latency * 1e6))
    if(plabel==int(label)):
        correct = correct+1
    print("Predicted Label: ", labels[plabel])
print("Number of Files Tested: ", nfiles)
print("Accuracy: ", 100*(correct/nfiles))

"""
lenet_array = lenet_mmio.array    # numpy array of uint32
print("array size = {}".format(lenet_array.size))

gg=0
lenet_mmio.write(OFFSET_G_COUNT, gg)
lenet_mmio.write(OFFSET_CTRL, 0x01)
ggg=lenet_mmio.read(OFFSET_GG_COUNT)
r=lenet_mmio.read(OFFSET_R_COUNT)
print("start: gg,ggg,r:",gg,ggg,r)

for fname in ['./lenet/1.csv', './lenet/2.csv']:
# for fname in ['./lenet/1.csv']:
	x = []
	y = []
	z = []
	count = 0
	for line in open(fname).readlines():
		sline = line.split(',')
		x.append(float(sline[0]))
		y.append(float(sline[1]))
		z.append(float(sline[2]))
		count = count+1
	print("count:",gg,count)
	lenet_mmio.write(OFFSET_COUNT,count)
	print("writing data:")
	i = 0
	while i<count:
		tmp = x[i]
#		print("x", tmp)
		bdata=struct.pack('>d',tmp)
		upper = int.from_bytes(bdata[0:4],'big')
		lower = int.from_bytes(bdata[4:8],'big')
		lenet_mmio.write(OFFSET_INDATA+i*6*4, lower)
		lenet_mmio.write(OFFSET_INDATA+(i*6+1)*4, upper)
		tmp = y[i]
#		print("y", tmp)
		bdata=struct.pack('>d',tmp)
		upper = int.from_bytes(bdata[0:4],'big')
		lower = int.from_bytes(bdata[4:8],'big')
		lenet_mmio.write(OFFSET_INDATA+(i*6+2)*4, lower)
		lenet_mmio.write(OFFSET_INDATA+(i*6+3)*4, upper)
		tmp = z[i]
#		print("z", tmp)
		bdata=struct.pack('>d',tmp)
		upper = int.from_bytes(bdata[0:4],'big')
		lower = int.from_bytes(bdata[4:8],'big')
		lenet_mmio.write(OFFSET_INDATA+(i*6+4)*4, lower)
		lenet_mmio.write(OFFSET_INDATA+(i*6+5)*4, upper)
		i = i+1
	gg=gg+1

	print("go:")
	lenet_mmio.write(OFFSET_G_COUNT, gg)
	while 1 :
		time.sleep(1)
		ggg=lenet_mmio.read(OFFSET_GG_COUNT)
		r=lenet_mmio.read(OFFSET_R_COUNT)
		print("wait for the result:gg,ggg,r:",gg,ggg,r)
#		if ggg==gg:
		if gg+1==r:
			break

	max = lenet_mmio.read(OFFSET_MAX_COUNT)
	print ("max:",max)
#	rr=0
#	r=[]
#	while rr<8:
#		itmpl = lenet_mmio.read(OFFSET_BOUT+rr*4)
#		bdata1=struct.pack('>I',itmpl)
#		itmpu = lenet_mmio.read(OFFSET_BOUT+(rr+1)*4)
#		bdata2=struct.pack('>I',itmpu)
#		bdata3 = bdata2+bdata1
#		result = struct.unpack('>d',bdata3)
#		print ("result:",result[0])
#		r.append(result[0])
#		rr=rr+2
while 1 :
	status = lenet_mmio.read(OFFSET_CTRL)
	print("status:", status)
	if((status&0x2)==0x2): break        
	time.sleep(1)
rr=0
r=[]
while rr<8:
	itmpl = lenet_mmio.read(OFFSET_BOUT+rr*4)
	bdata1=struct.pack('>I',itmpl)
	itmpu = lenet_mmio.read(OFFSET_BOUT+(rr+1)*4)
	bdata2=struct.pack('>I',itmpu)
	bdata3 = bdata2+bdata1
	result = struct.unpack('>d',bdata3)
	print ("result:",result[0])
	r.append(result[0])
	rr=rr+2
"""
