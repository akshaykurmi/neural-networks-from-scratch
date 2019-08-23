import numpy as np
from .base import Layer


class Convolution2D(Layer):
    def __init__(self, num_filters, filter_size, stride, padding, activation, weights_initializer, bias_initializer,
                 input_shape=None):
        super().__init__()
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.input_shape = input_shape
        self.activation = activation
        self.bias_initializer = bias_initializer
        self.weights_initializer = weights_initializer
        self.W, self.dW, self.b, self.db = None, None, None, None
        self.cache = {}

    def _initialize_parameters(self):
        batch_size, channels, height, width = self.input_shape
        filter_height, filter_width = self.filter_size
        assert (height + 2 * self.padding - filter_height) % self.stride == 0
        assert (width + 2 * self.padding - filter_width) % self.stride == 0
        output_height = 1 + (height + 2 * self.padding - filter_height) // self.stride
        output_width = 1 + (width + 2 * self.padding - filter_width) // self.stride
        self.output_shape = (batch_size, self.num_filters, output_height, output_width)
        self.W = self.weights_initializer.initialize((self.num_filters, channels, filter_height, filter_width))
        self.b = self.bias_initializer.initialize((self.num_filters,))

    def forward(self, X, *args, **kwargs):
        _, _, output_height, output_width = self.output_shape
        batch_size = self.input_shape[0]
        X_cols = self._im2col(X)
        output = self.W.reshape((self.num_filters, -1)).dot(X_cols) + self.b.reshape(-1, 1)
        output = output.reshape(self.num_filters, output_height, output_width, batch_size)
        output = output.transpose(3, 0, 1, 2)
        self.cache = {"X_cols": X_cols}
        return self.activation.compute(output)

    def backward(self, gradients):
        gradients = gradients * self.activation.gradients()
        self.db = np.sum(gradients, axis=(0, 2, 3))
        gradients = gradients.transpose(1, 2, 3, 0).reshape(self.num_filters, -1)
        self.dW = gradients.dot(self.cache["X_cols"].T).reshape(self.W.shape)
        dX_cols = self.W.reshape(self.num_filters, -1).T.dot(gradients)
        return self._col2im(dX_cols)

    def _im2col(self, im):
        filter_height, filter_width = self.filter_size
        channels = self.input_shape[1]
        pad_width = ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding))
        padded = np.pad(im, pad_width, mode='constant')
        i, j, k = self._im2col_indices()
        cols = padded[:, k, i, j]
        return cols.transpose(1, 2, 0).reshape(filter_height * filter_width * channels, -1)

    def _col2im(self, cols):
        filter_height, filter_width = self.filter_size
        batch_size, channels, height, width = self.input_shape
        padded_height, padded_width = height + 2 * self.padding, width + 2 * self.padding
        i, j, k = self._im2col_indices()
        cols = cols.reshape(channels * filter_height * filter_width, -1, batch_size).transpose(2, 0, 1)
        im = np.zeros((batch_size, channels, padded_height, padded_width), dtype=cols.dtype)
        np.add.at(im, (slice(None), k, i, j), cols)
        if self.padding == 0:
            return im
        return im[:, :, self.padding:-self.padding, self.padding:-self.padding]

    def _im2col_indices(self):
        filter_height, filter_width = self.filter_size
        output_height, output_width = self.output_shape[-2:]
        channels = self.input_shape[1]
        i0 = np.tile(np.repeat(np.arange(filter_height), filter_width), channels)
        i1 = self.stride * np.repeat(np.arange(output_height), output_width)
        j0 = np.tile(np.arange(filter_width), filter_height * channels)
        j1 = self.stride * np.tile(np.arange(output_width), output_height)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)
        k = np.repeat(np.arange(channels), filter_height * filter_width).reshape(-1, 1)
        return i, j, k

    @property
    def parameters_and_gradients(self):
        return [(self.W, self.dW), (self.b, self.db)]

    @property
    def num_params(self):
        return np.prod((*self.W.shape, *self.b.shape))
