#! /usr/bin/python
# -*- coding: utf-8 -*-
#
import os
import sys
import time
import pickle
import numpy as np

import core
import opencl
import mnist

def pickle_load(path):
    try:
        with open(path, mode='rb') as f:
            data = pickle.load(f)
            return data
        #
    except:
        return None
    #
    
def print_result(ca, eval_size, num_class, dist, rets, oks):
    print("---------------------------------")
    print(("result : %d / %d" % (ca, eval_size)))
    accuracy = float(ca) / float(eval_size)
    print(("accuracy : %f" % (accuracy)))
    print("---------------------------------")
    print("class\t|dist\t|infs\t|ok")
    print("---------------------------------")
    for i in range(num_class):
        print(("%d\t| %d\t| %d\t| %d"  % (i, dist[i], rets[i], oks[i])))
    #
    print("---------------------------------")
    
def classification(r, data_size, num_class, batch_size, batch_image, batch_label, n, debug=0, single=0):
    
    dist = np.zeros(num_class, dtype=np.int32)
    rets = np.zeros(num_class, dtype=np.int32)
    oks = np.zeros(num_class, dtype=np.int32)
    print((">>test(%d) = %d" % (n, batch_size)))
    print(num_class)
    it, left = divmod(batch_size, n)
    
    # for single test
    if single==1:
        it = 1
        n = 1
    #
    
    if left>0:
        print(("error : n(=%d) is not appropriate" % (n)))
    #
    #start_time = time.time()
    elapsed_time = 0.0
    #
    r.prepare(n, data_size, num_class)
    data_array = np.zeros((n, data_size), dtype=np.float32)
    class_array = np.zeros(n, dtype=np.int32)
    for i in range(it):
        for j in range(n):
            data_array[j] = batch_image[i*n+j]
            class_array[j] = batch_label[i*n+j]
        #
        r.set_batch(data_size, num_class, data_array, class_array, n, 0)
        start_time = time.time()
        r.propagate(debug)
        elapsed_time += (time.time() - start_time)
        #
        #infs = r.get_inference()
        answers = r.get_answer()
        #print(answers)
        for j in range(n):
            ans = answers[j]
            label = class_array[j]
            rets[ans] = rets[ans] + 1
            dist[label] = dist[label] + 1
            #print("%d, %d" % (ans, label))
            if ans == label:
                oks[ans] = oks[ans] + 1
            #
        #
    #
    ca = sum(oks)
    print_result(ca, batch_size, num_class, dist, rets, oks)
    #
    #elapsed_time = time.time() - start_time
    t = format(elapsed_time, "0")
    print(("time = %s" % (t)))
    print("done")
    return float(ca) / float(batch_size)
    
def main():
    argvs = sys.argv
    argc = len(argvs)
    print(argvs)
    print(argc)
    
    batch_offset = 0
    data_size = mnist.IMAGE_SIZE
    num_class = mnist.NUM_CLASS
    
    platform_id = 0
    device_id = 1
    my_gpu = opencl.OpenCL(platform_id, device_id)
    my_gpu.set_kernel_code()

    r = mnist.setup_dnn(my_gpu, "./wi-fc.csv")

    #if config==0:
    #    r = mnist.setup_dnn(my_gpu, config, "./wi-fc.csv")
    #elif config==1:
    #    r = mnist.setup_dnn(my_gpu, config, "./wi-cnn.csv")
    #else:
    #    return 0
    #
    if r:
        pass
    else:
        return 0
    #
    #r.set_gpu()
    
    #r.prepare(100, data_size, num_class)
    #r.propagate()
    
    #return 0
    
    batch_size = mnist.TEST_BATCH_SIZE
    batch_image = pickle_load(mnist.TEST_IMAGE_BATCH_PATH)
    batch_label = pickle_load(mnist.TEST_LABEL_BATCH_PATH)
        
    ac = classification(r, data_size, num_class, batch_size, batch_image, batch_label, 100)#, 1, 1)
    print(ac)
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
