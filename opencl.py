#! /usr/bin/python
# -*- coding: utf-8 -*-
#
import os
import sys
import numpy as np
import pyopencl as cl

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

KERNEL_CODE = """
__kernel void scale(
    __global float* x,
    __global float* y,
    const int stride,
    const float max,
    const int debug)
{
    int i = get_global_id(0); // data index
    int j = get_global_id(1); // bathch index
    /*
    if (x[stride*j+i]==0){
        y[stride*j+i] = 0.0001;
    }else{
        y[stride*j+i] = x[stride*j+i]/max;
    }
    */
    y[stride*j+i] = x[stride*j+i]/max;
    
    if (debug==1){
        printf(\"%d, %d\\n\",i, j);
    }
};

__kernel void softmax(__global float* in, int num)
{
    int bi = get_global_id(0);
    float temp = 0.0;
    float total = 0.0;
    int start = bi*num;

    for (int i=0;i<num;i++){
        temp = in[start+i];
        temp = exp(temp);
        if (isinf(temp)){
            temp = 3.402823e+38;
        }else if (isnan(temp)){
            temp = 0;
        }
        in[start+i] = temp;
        total += temp;
    }

    for (int i=0;i<num;i++){
        //printf(\"%d | %f : %f\\n\", i, in[start+i], total);
        in[start+i] = in[start+i]/total;
    }
}

__kernel void calc_mac_relu(
    __global float* x,
    __global float* w,
    __global float* y,
    int xsize, // node
    int wsize, // input
    int act)
{
    int bi = get_global_id(0); // batch
    int xi = get_global_id(1); // node
    
    int x_start = wsize * bi;
    int w_start = wsize * xi;
    int y_start = (xsize * bi) + xi;
    float temp = 0.0;

    for (int i=0;i<wsize;i++){
        temp += (x[x_start+i] * w[w_start+i]);
    }

    // activation
    if (temp>=0){
        y[y_start] = temp;
    }else{
        if (act==0){ // no
            y[y_start] = temp;
        }else if (act==1){ // relu
            y[y_start] = 0;
        }else if (act==2){
            y[y_start] = 0.000001;
        }else if (act==3){
            y[y_start] = temp/20;
        }else{
            y[y_start] = temp;
        }
    }
}

__kernel void k_test(const float in)
{
    int i = get_global_id(0);
    float out = 0.0;
    out = exp(in);
    printf(\"%d : exp(%If) = %If\\n\", i, in, out);
};
"""
#
#
#
class OpenCL(object):
    def __init__(self, platform_id, device_id):
        self.name = "OpenCL"
        self.platform_id = platform_id
        self.device_id = device_id
        platform = cl.get_platforms()[platform_id]
        device = platform.get_devices()[device_id]
        print(platform)
        print(device)
        #
        self._ctx = cl.Context([device])
        for dev in self._ctx.devices:
            assert dev.local_mem_size > 0
        #
        self._queue = cl.CommandQueue(self._ctx)
        self._bufs = []

    def set_kernel_code(self):
        self.prg = cl.Program(self._ctx, KERNEL_CODE).build()
    
    def get_buffer_list(self):
        return self._bufs
    
    def dev_malloc(self, host_array):
        mf = cl.mem_flags
        buf = cl.Buffer(self._ctx,
                        mf.READ_WRITE|mf.COPY_HOST_PTR,
                        hostbuf=host_array,
                        size=host_array.nbytes)
        self._bufs.append(buf)
        return buf
        
    def copy(self, dist, src):
        event = cl.enqueue_copy(self._queue, dist, src)
        event.wait()

    #
    #
    #
    
    def scale(self, d_x, d_y, stride, max, row, batch_size, debug):
        event = self.prg.scale(self._queue, (row, batch_size), None,
                               d_x, d_y, np.int32(stride),
                               np.float32(max), np.int32(debug))
        event.wait()
        
    def macRelu(self, buf_x, buf_w, buf_y, size_batch, size_node, size_input, act):
        event = self.prg.calc_mac_relu(self._queue,(size_batch, size_node), None,
                                        buf_x, buf_w, buf_y,
                                        np.int32(size_node), np.int32(size_input),
                                        np.int32(act))
        event.wait()
    
    def softmax(self, data, size, num_batch): # <<
        event = self.prg.softmax(self._queue, (num_batch,), None, data, np.int32(size))
        event.wait()

    def k_test(self, value):
        event = self.prg.k_test(self._queue, (1,), None, np.float32(value))
        event.wait()
#
#
#
def main():
    data_x = np.array([0.1, 0.2, 0.3, 0.4, 0.5]).astype(np.float32)
    data_w = np.array([[0.5, 0.5, 0.5, 0.5, 0.2, 0.5, 0.5, 0.5, 0.5, 0.2, 0.5, 0.5, 0.5, 0.5, 0.2],
                       [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                       [0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5]]).astype(np.float32)
    data_y = np.array([[0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0]]).astype(np.float32)
    data_a = np.array([8, 16, 32, 64]).astype(np.int32)
    data_b = np.array([0.0, 0.0, 0.0, 0.0]).astype(np.float64)

    print(data_x)
    print(data_w)
    print(data_y)
    
    platform_id = 0
    device_id = 1
    g = Gpu(platform_id, device_id)
    g.set_kernel_code()
    #
    p = 0.0
    for i in range(100):
        g.k_test(p)
        p = p + 1.0
    #
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

