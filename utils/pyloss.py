import caffe
import numpy as np


class MMDLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match. bottom[0]: (n, c); bottom[1]: (n, c).
        if bottom[0].data.shape != bottom[1].data.shape:
            raise Exception("Inputs must have the same dimension.")
        self._dimentions = bottom[0].data.shape[1]
        # difference is shape of inputs
        self.diff = np.zeros((self._dimentions), dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        data_mean0 = np.mean(bottom[0].data, axis=0)
        data_mean1 = np.mean(bottom[1].data, axis=0)
        self.diff[...] = data_mean0 - data_mean1
        top[0].data[...] = np.sum(self.diff**2) / self._dimentions / 2.

    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff / self._dimentions
