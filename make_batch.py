#! /usr/bin/python
# -*- coding: utf-8 -*-
#
import os
import sys
import time
import numpy as np
import pickle

import mnist

def pickle_save(path, data):
    with open(path, mode='wb') as f:
        pickle.dump(data, f)
    #

def make_batch(batch_size,
               image_path_in, label_path_in,
               image_path_out, label_path_out):
    file_in = open(label_path_in, 'rb')
    header = file_in.read(mnist.LABEL_HEADER_SIZE)
    data = file_in.read()
    #
    labels = [0 for i in range(batch_size)] # list
    #
    for i in range(batch_size):
        if i<10:
            print(data[i])
        #
        #label = struct.unpack('>B', data[i])
        labels[i] = int(data[i])
    #
    file_in = open(image_path_in, 'rb')
    header = file_in.read(mnist.IMAGE_HEADER_SIZE)
    #
    images = np.zeros((batch_size, (mnist.IMAGE_SIZE)), dtype=np.float32)
    #
    for i in range(batch_size):
        image = file_in.read(mnist.IMAGE_SIZE)
        da = np.frombuffer(image, dtype=np.uint8)
        a_float = da.astype(np.float32) # convert from uint8 to float32
        images[i] = a_float
    #
    print((len(labels)))
    print((images.shape[0]))
    
    pickle_save(image_path_out, images)
    pickle_save(label_path_out, labels)

def main():
    argvs = sys.argv
    argc = len(argvs)

    make_batch(mnist.TEST_BATCH_SIZE,
               mnist.TEST_IMAGE_PATH,  mnist.TEST_LABEL_PATH,
               mnist.TEST_IMAGE_BATCH_PATH, mnist.TEST_LABEL_BATCH_PATH)
    
    return 0
#
#
#
if __name__=='__main__':
    print(">> start")
    sts = main()
    print(">> end")
    print("\007")
    sys.exit(sts)
#
#
#
