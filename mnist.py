#! /usr/bin/python
# -*- coding: utf-8 -*-
#

import os
import sys

import core

IMAGE_HEADER_SIZE = 16
LABEL_HEADER_SIZE  = 8
IMAGE_SIZE = 784
NUM_CLASS = 10

DATA_BASE_PATH = "./data/MNIST/raw/"
TRAIN_IMAGE_PATH = DATA_BASE_PATH + "train-images-idx3-ubyte"
TRAIN_LABEL_PATH = DATA_BASE_PATH + "train-labels-idx1-ubyte"
TEST_IMAGE_PATH = DATA_BASE_PATH + "t10k-images-idx3-ubyte"
TEST_LABEL_PATH = DATA_BASE_PATH + "t10k-labels-idx1-ubyte"

BATCH_BASE_PATH = "./batch/"

TRAIN_BATCH_SIZE = 60000
TRAIN_IMAGE_BATCH_PATH = BATCH_BASE_PATH + "train_image_batch.pickle"
TRAIN_LABEL_BATCH_PATH = BATCH_BASE_PATH + "train_label_batch.pickle"

TEST_BATCH_SIZE = 10000
TEST_IMAGE_BATCH_PATH = BATCH_BASE_PATH + "test_image_batch.pickle"
TEST_LABEL_BATCH_PATH = BATCH_BASE_PATH + "test_label_batch.pickle"

def setup_dnn(my_gpu, path):
    r = core.Roster()
    r.set_gpu(my_gpu)
    
    c = r.count_layers()
    input = core.InputLayer(c, IMAGE_SIZE, IMAGE_SIZE, None, r._gpu)
    r.layers.append(input)
    # 1 : hidden
    c = r.count_layers()
    hidden_1 = core.HiddenLayer(c, IMAGE_SIZE, 512, input, r._gpu)
    r.layers.append(hidden_1)
    # 2 : hidden
    c = r.count_layers()
    hidden_2 = core.HiddenLayer(c, 512, 256, hidden_1, r._gpu)
    r.layers.append(hidden_2)
    # 3 : output
    c = r.count_layers()
    output = core.OutputLayer(c, 256, 10, hidden_2, r._gpu)
    r.layers.append(output)
    
    r.set_path(path)
    #r.set_scale_input(1)
    r.load()
    r.update_weight()
    return r

