#! c:/Python26/python.exe
# -*- coding: utf-8 -*-

import os, sys, time, math
from stat import *
import random
import copy
#import multiprocessing as mp
import pickle
import numpy as np

if sys.platform.startswith('darwin'):
    pass
#else:
#    import plat
#    if plat.ID==2:
#        import cupy as cp
#        import cupyx
#    #
#
import csv
from PIL import Image

# LDNN Modules
#import gpu
#import util
#
#
#
LAYER_TYPE_INPUT   = 0
LAYER_TYPE_HIDDEN  = 1
LAYER_TYPE_OUTPUT  = 2
LAYER_TYPE_CONV    = 3
LAYER_TYPE_MAX     = 4
#
class Layer(object):
    # i         : index of layers
    # type      : 0 = input, 1 = hidden, 2 = output
    # input : stimulus from a previous layer
    # num_input : number of inputs / outputs from a previous layer
    # node : neurons
    # num_node  : numbers of neurons in a layer
    def __init__(self, i, type, num_input, num_node, pre, gpu=None):
        self._pre = None
        self._next = None
        self._pre = pre
        if self._pre:
            self._pre._next = self
        #
        self._index = i
        self._type = type
        self._gpu = gpu        
        self._id = -1
        self._num_input = num_input
        self._num_node = num_node
        
    def count_weight(self):
        return self._num_node*self._num_input
        
    def get_pre_layer(self):
        return self._pre
        
    def prepare(self, batch_size):
        pass
    
    def get_num_node(self):
        return self._num_node
        
    def get_num_input(self):
        return self._num_input
    
    # gpu must be checked before this method is called
    def update_weight(self):
        pass

    def propagate(self, array_in, debug=0):
        pass

    def set_weight(self, ni, ii, w):
        self._weight_matrix[ni][ii] = w

    def init_all_weight(self):
        for ni in range(self._num_node):
            for ii in range(self._num_input):
                w = 0
                self.set_weight(ni, ii, w)
            #
        #
    
    def import_weight(self, wi_list):
        self._weight_matrix = np.array(wi_list, dtype=np.float32).copy()
        
    def set_id(self, id):
        if id>=0:
            self._id = id

    def get_id(self):
        return self._id
    
    def get_type(self):
        return self._type
        
    def reset(self):
        pass
#
#
#
class InputLayer(Layer):
    def __init__(self, i, num_input, num_node, pre, gpu=None):
        print("InputLayer::__init__()")
        super(InputLayer, self).__init__(i, LAYER_TYPE_INPUT, num_input, num_node, pre, gpu)
        
    def prepare(self, batch_size):
        self._batch_size = batch_size
        self._output_array = np.zeros((self._batch_size, self._num_node), dtype=np.float32)
        if self._gpu:
            self._gpu_output = self._gpu.dev_malloc(self._output_array)
        #

    def propagate(self, array_in, debug=0):
        if debug:
            print("input")
            if self._gpu: # OpenCL
                self._gpu.copy(self._output_array, self._gpu_output)
                print((self._output_array[0]))
            #
        #
    
    def set_weight_index(self, ni, ii, wi):
        pass
        
    def get_weight_index(self, ni, ii):
        return 0
        
    def export_weight_index(self):
        return None
        
    def count_weight(self):
        return 0

class HiddenLayer(Layer):
    def __init__(self, i, num_input, num_node, pre, gpu=None):
        print("HiddenLayer::__init__()")
        super(HiddenLayer, self).__init__(i, LAYER_TYPE_HIDDEN, num_input, num_node, pre, gpu)
    
        self._weight_matrix = np.zeros( (self._num_node, self._num_input), dtype=np.float32)
        
        if self._gpu:
            self._gpu_weight = self._gpu.dev_malloc(self._weight_matrix)
        else:
            print("error")
        #
    
    def prepare(self, batch_size):
        print("HiddenLayer::prepare(%d)" % (batch_size))
        self._batch_size = batch_size
        self._output_array = np.zeros((self._batch_size, self._num_node), dtype=np.float32)
        
        if self._gpu:
            self._gpu_output = self._gpu.dev_malloc(self._output_array)
            self.update_weight()
        else:
            pass
        #
        print(self._weight_matrix.shape)

    def update_weight(self):
        if self._gpu:
            self._gpu.copy(self._gpu_weight, self._weight_matrix)
        else:
            pass
        #

    def propagate(self, array_in, debug=0):
        if self._gpu:
            pass
        else:
            return
        #
        
        stride_1 = self._num_node * self._num_input
        stride_2 = self._num_input
        # activation mode
        #   0 : none
        #   1 : normal
        #   2 : 0.000001
        #   3 : y/20
        a_mode = 1
        
        self._gpu.macRelu(array_in, self._gpu_weight, self._gpu_output, self._batch_size, self._num_node, self._num_input, a_mode)
        
        if debug:
            #print("scale", self._scale)
            print(self._index, "hidden, input")
            tarray = np.zeros((self._batch_size, self._num_input), dtype=np.float32)
            self._gpu.copy(tarray, array_in)
            print((tarray[0]))
                    
            print(self._index, "hidden")
            self._gpu.copy(self._output_array, self._gpu_output)
            print((self._output_array[0]))
        #

class OutputLayer(Layer):
    def __init__(self, i, num_input, num_node, pre, gpu=None):#, mode=0):
        print("OutputLayer::__init__()")
        super(OutputLayer, self).__init__(i, LAYER_TYPE_OUTPUT, num_input, num_node, pre, gpu)#, mode)
        #
        self._weight_matrix = np.zeros( (self._num_node, self._num_input), dtype=np.float32)
        if self._gpu:
            self._gpu_weight = self._gpu.dev_malloc(self._weight_matrix)
        #

    def prepare(self, batch_size):
        print("OutputLayer::prepare(%d)" % (batch_size))
            
        self._batch_size = batch_size
        self._output_array = np.zeros((self._batch_size, self._num_node), dtype=np.float32)
        self._softmax_array = np.zeros((self._batch_size, self._num_node), dtype=np.float64)
        
        if self._gpu:
            self._gpu_output = self._gpu.dev_malloc(self._output_array)
            self._gpu_softmax = self._gpu.dev_malloc(self._softmax_array)
            
            self.update_weight()
        else:
            pass
        #
        
    def update_weight(self):
        if self._gpu:
            self._gpu.copy(self._gpu_weight, self._weight_matrix)
        else:
            pass
        #
        
    def propagate(self, array_in, debug=0):
        stride_1 = self._num_node * self._num_input
        stride_2 = self._num_input
        activation = 1

        if self._gpu:
            self._gpu.macRelu(array_in, self._gpu_weight, self._gpu_output, self._batch_size, self._num_node, self._num_input, 0)
            # softmax
            if debug:
                print("output")
                self._gpu.copy(self._output_array, self._gpu_output)
                print((self._output_array[0]))
            #
            self._gpu.softmax(self._gpu_output, self._num_node, self._batch_size)
            if debug:
                print("softmax")
                self._gpu.copy(self._output_array, self._gpu_output)
                print((self._output_array[0]))
            #
        else:
            pass
        #
        
class Roster:
    def __init__(self):
        self._weight_list = []
        self._gpu = None
        self.layers = []
        self.input = None
        self.output = None
        self._batch_size = 1
        self._data_size = 1
        self._eval_mode = 0
        self._path = ""
        self._scale_input = 1
        #self.mem_save = 1
        
    def set_path(self, path):
        self._path = path
        
    def load(self):
        print("Roster::load(%s)" % (self._path))
        if os.path.isfile(self._path):
            self.import_weight(self._path)
        else:
            mode = 1
            value = 0
            self.init_weight(mode, value)
            #self.export_weight(self._path)
        #
    
    def set_gpu(self, gpu):
        self._gpu = gpu

    def prepare(self, batch_size, data_size, num_class):
        self.num_class = num_class
        self._batch_size = batch_size
        self._data_size = data_size
        #
        self._batch_data = np.zeros((self._batch_size, data_size), dtype=np.float32)
        self._labels = np.zeros((batch_size, num_class), dtype=np.float32)
        #
        if self._gpu: # OpenCL
            self._batch_cross_entropy = np.zeros(batch_size, dtype=np.float32)
            self._gpu_input = self._gpu.dev_malloc(self._batch_data)
            self._gpu_labels = self._gpu.dev_malloc(self._labels)
            self._gpu_entropy = self._gpu.dev_malloc(self._batch_cross_entropy)
        #
        self.input = self.get_layer_at(0)
        for layer in self.layers:
            layer.prepare(batch_size)
        #
        self.output = layer
    
    # batch for classification
    #def set_batch(self, data_size, num_class, train_data_batch, train_label_batch, size, offset):
    def set_batch(self, data_size, num_class, data_batch, label_batch, size, offset):
        print("Roster::set_batch(%d, %d, %d, %d)" % (data_size, num_class, size, offset))
        
        data_array = np.zeros((size, data_size), dtype=np.float32)
        labels = np.zeros((size, num_class), dtype=np.float32)
        for j in range(size):
            data_array[j] = data_batch[offset+j]
            k = int(label_batch[offset+j])
            #print(k)
            labels[j][k] = 1.0
        #
        self.set_data(data_array, data_size, labels, size)
                
    def set_data(self, data, data_size, label, batch_size):
        #print("Roster::set_data(%d, %d)" % (data_size, batch_size))
        if self._gpu:
            pass
        else:
            return
        #
        
        self.reset()
        
        self._gpu.copy(self._gpu_input, data)
        self._gpu.copy(self._gpu_labels, label)
        self._gpu.scale(self._gpu_input, self.input._gpu_output, data_size, float(255.0), self.input._num_node, batch_size, 0)
    
    def set_batch_data(self, data_size, train_data_batch, size, offset, scale=0):
        print("Roster::set_batch_data(%d, %d, %d, %d)" % (data_size, size, offset, scale))
        
        data_array = np.zeros((size, data_size), dtype=np.float32)
        for j in range(size):
            data_array[j] = train_data_batch[offset+j]
        #
        self.reset()
        
        self._gpu.copy(self._gpu_input, data_array)

        if self._scale_input==0:
            self._gpu.copy(self.input._gpu_output, self._gpu_input)
        elif self._scale_input==1:
            x_gpu = self._gpu.allocateArray(self._gpu_input)
            y_gpu = x_gpu / 255.0
            self.output._gpu_output = self._gpu.allocateArray(y_gpu)
        elif self._scale_input==2:
            self._gpu.scale_exp(self._gpu_input, self.input._gpu_output, data_size, self.input._num_node, size, 0)
        else:
            pass
        #
        
    def set_batch_label(self, data_size, train_label_batch, size, offset, scale=0):
        print("Roster::set_batch_label(%d, %d, %d, %d)" % (data_size, size, offset, scale))
        labels = np.zeros((size, data_size), dtype=np.float32)
        for j in range(size):
            labels[j] = train_label_batch[offset+j]
        #
        self.reset()
        self._gpu.copy(self._gpu_labels, labels)
    
    def direct_set_data(self, data_array):
        self._gpu.copy(self._gpu_input, data_array)
        self._gpu.copy(self.input._gpu_output, self._gpu_input)
        
    def direct_set_label(self, label_array):
        self._gpu.copy(self._gpu_labels, label_array)
    
    def init_weight(self, mode=0, value=0):
        c = self.count_layers()
        for i in range(c):
            layer = self.get_layer_at(i)
            type = layer.get_type()
            if type==LAYER_TYPE_INPUT:
                pass
            else:
                #layer.init_weight_with_random_index()
                #layer.init_weight_with_mode(mode, value)
                layer.init_all_weight()
            #
        #
        
    def init_weight_by_layer(self, idx, mode, value=0):
        layer = self.get_layer_at(idx)
        if layer.count_weight()>0:
            if mode==0: # random
                layer.init_weight_with_random_index()
            elif mode==1: # a value
                layer.init_weight_with_value(value)
            #
        #
        
    def reset(self):
    # flush a batch depending cache when switching batches
        c = self.count_layers()
        for i in range(c):
            layer = self.get_layer_at(i)
            layer.reset()
        #
        
    def count_weight(self):
        cnt = 0
        c = self.count_layers()
        for i in range(1, c):
            layer = self.get_layer_at(i)
            cnt = cnt + layer.count_weight()
        #
        return cnt

    def update_weight(self):
        for layer in self.layers:
            layer.update_weight()
        #

    def count_layers(self):
        return len(self.layers)

    def get_layers(self):
        if self.count_layers() == 0:
            return 0
        #
        return self.layers
    
    def get_layer_at(self, i):
        c = self.count_layers()
        if i>=c:
            print("error : Roster : get_layer_at")
            return None
        #
        return self.layers[i]

    def add_layer(self, type, num_input, num_node):
        c = self.count_layers()
        if type==LAYER_TYPE_INPUT:
            layer = InputLayer(c, num_input, num_node, self._gpu)
            self.layers.append(layer)
            return layer
        elif type==LAYER_TYPE_HIDDEN:
            layer = HiddenLayer(c, num_input, num_node, self._gpu)
            self.layers.append(layer)
            return layer
        elif type==LAYER_TYPE_OUTPUT:
            layer = OutputLayer(c, num_input, num_node, self._gpu)
            self.layers.append(layer)
            return layer
        else:
            print("unknown type %d" % (LAYER_TYPE_OUTPUT))
            return
 
    def get_inference(self):
        c = self.count_layers()
        output = self.get_layer_at(c-1)
        output._gpu.copy(output._output_array, output._gpu_output)
        return output._output_array

    def get_answer(self):
        #print("roster::get_answer()")
        ret = []
        output = self.output
        if self._gpu:
            output._gpu.copy(output._output_array, output._gpu_output)
        else:
            print("core::get_answer() = error")
            return ret
        #

        for i in range(self._batch_size):
            inf = output._output_array[i]
            
            max_index = -1
            max = -1.0
            for j in range(self.num_class):
                if inf[j]>max:
                    max = inf[j]
                    max_index = j
                #
            #
            ret.append(max_index)
        #
        return ret
        
    def import_weight(self, path):
        print(("Roster::import_weight(%s)" % path))
        
        with open(path, "r") as f:
            reader = csv.reader(f)
            lc = self.count_layers()
            for i in range(1, lc):
                layer = self.get_layer_at(i)
                type = layer.get_type()
                if type==LAYER_TYPE_INPUT or type==LAYER_TYPE_MAX:
                    continue
                #
                nc  = layer._num_node
                block = []
                for row in reader:
                    line = []
                    for cell in row:
                        line.append(cell)
                    #
                    block.append(line)
                    if len(block)==nc:
                        break
                    #
                #
                layer.import_weight(block)
            # for
        # with

    def propagate(self, debug=0):
        c = self.count_layers()
        pre = self.get_layer_at(0)
        for i in range(1, c):
            #print(i)
            layer = self.get_layer_at(i)
            layer.propagate(pre._gpu_output, debug)
            #
            pre = layer
        #
        #print("end of propagate()")
#
#
#
def main():
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
# EOF
