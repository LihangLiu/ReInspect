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
        # print 'bottom', bottom[0].data[0,:5]
        # print 'bottom', bottom[1].data[0,:5]
        # print 'mean', data_mean0[:5]
        # print 'diff', self.diff[:5]
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


class MMDClass0LossLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 4:
            raise Exception("Need four inputs to compute binary class loss.")

    def reshape(self, bottom, top):
        # check input dimensions match. bottom[0]: (n, c); bottom[1]: (n, c)
        #                               bottom[2]: (n, 2); bottom[3]: (n, 2);
        if bottom[0].data.shape != bottom[1].data.shape:
            raise Exception("Inputs must have the same dimension.")
        if bottom[2].data.shape != bottom[3].data.shape:
            raise Exception("Inputs must have the same dimension.")
        if bottom[2].data.shape[1] != 2:
  
            raise Exception("Inputs must have the same dimension of 2")
        self._dimentions = bottom[0].data.shape[1]
        self._n = bottom[0].data.shape[0]
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        y0 = bottom[2].data[:,0,0,0]
        y1 = bottom[3].data[:,0,0,0]
        data_mean0 = np.dot(y0, bottom[0].data) / np.sum(self._n)
        data_mean1 = np.dot(y1, bottom[1].data) / np.sum(self._n)
        self.diff[...] = np.dot(np.reshape(y0, (self._n,1)), 
                                np.reshape(data_mean0-data_mean1, (1,self._dimentions)))
        top[0].data[...] = np.sum((data_mean0 - data_mean1)**2) / self._dimentions / 2.

    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff / (self._dimentions*self._n)


class MMDClass1LossLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 4:
            raise Exception("Need four inputs to compute binary class loss.")

    def reshape(self, bottom, top):
        # check input dimensions match. bottom[0]: (n, c); bottom[1]: (n, c)
        #                               bottom[2]: (n, 2); bottom[3]: (n, 2);
        if bottom[0].data.shape != bottom[1].data.shape:
            raise Exception("Inputs must have the same dimension.")
        if bottom[2].data.shape != bottom[3].data.shape:
            raise Exception("Inputs must have the same dimension.")
        if bottom[2].data.shape[1] != 2:
            raise Exception("Inputs must have the same dimension of 2")
        self._dimentions = bottom[0].data.shape[1]
        self._n = bottom[0].data.shape[0]
        # difference is shape of inputs
        self.diff = np.zeros((self._dimentions), dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        y0 = bottom[2].data[:,1,0,0]
        y1 = bottom[3].data[:,1,0,0]
        data_mean0 = np.dot(y0, bottom[0].data) / np.sum(y0)
        data_mean1 = np.dot(y1, bottom[1].data) / np.sum(y1)
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



class MMDVarLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match. bottom[0]: (n, c); bottom[1]: (n, c).
        if bottom[0].data.shape != bottom[1].data.shape:
            raise Exception("Inputs must have the same dimension.")
        self._n = bottom[0].data.shape[0]
        self._dimentions = bottom[0].data.shape[1]
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        data_mean0 = np.mean(bottom[0].data, axis=0)
        data_var0 = np.var(bottom[0].data, axis=0)
        data_var1 = np.var(bottom[1].data, axis=0)
        data_var1 = data_var1*np.linalg.norm(data_var0)/np.linalg.norm(data_var1)
        self.diff[...] = (data_var0 - data_var1) * (bottom[0].data - data_mean0)
        top[0].data[...] = np.sum((data_var0 - data_var1)**2) / self._dimentions / 2.

    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign*2*self.diff / (self._dimentions*self._n)   


class MMDParamLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 4:
            raise Exception("Need four inputs to compute binary class loss.")

    def reshape(self, bottom, top):
        # check input dimensions match. bottom[0]: (n, c); bottom[1]: (n, c)
        #                               bottom[2]: (c); bottom[3]: (c);
        if bottom[0].data.shape != bottom[1].data.shape:
            raise Exception("Inputs must have the same dimension.")
        if bottom[2].data.shape != bottom[3].data.shape:
            raise Exception("Inputs must have the same dimension.")
        if bottom[0].data.shape[1] != bottom[2].data.shape[0]:
            raise Exception("Inputs must have the same dimension of 2")
        self._dimentions = bottom[0].data.shape[1]
        self._n = bottom[0].data.shape[0]
        # difference is shape of inputs
        self.diff = np.zeros((self._dimentions), dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        p0 = 10*bottom[2].data                 # 100 is borrowed from HungarianLoss
        p1 = 10*bottom[3].data
        data_mean0 = p0 * np.mean(bottom[0].data, axis=0)
        data_mean1 = p1 * np.mean(bottom[1].data, axis=0)
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

