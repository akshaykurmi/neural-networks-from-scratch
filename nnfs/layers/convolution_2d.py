import numpy as np
from .base import Layer


class Convolution2D(Layer):
    def __init__(self, num_filters, kernel_size, strides, padding, activation, weights_initializer, bias_initializer,
                 input_shape=None):
        super().__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.input_shape = input_shape
        self.output_shape = None
        self.activation = activation
        self.bias_initializer = bias_initializer
        self.weights_initializer = weights_initializer
        self.W, self.dW, self.b, self.db = None, None, None, None
        self.most_recent_propagated_input = None

    def _initialize_parameters(self):
        batch_size, width, height, channels = self.input_shape
        output_width = 1 + (width + 2 * self.padding[0] - self.kernel_size[0]) // self.strides[0]
        output_height = 1 + (height + 2 * self.padding[1] - self.kernel_size[1]) // self.strides[1]
        self.output_shape = (batch_size, self.num_filters, output_width, output_height)
        self.W = self.weights_initializer.initialize((self.num_filters, *self.kernel_size, channels))
        self.b = self.bias_initializer.initialize((self.num_filters,))

    def make_first_layer(self):
        super().make_first_layer()
        assert self.input_shape is not None
        self._initialize_parameters()

    def connect_with(self, prev_layer):
        self.input_shape = prev_layer.output_shape
        self._initialize_parameters()

    def forward(self, propagated_input, *args, **kwargs):
        pad_widths = ((0, 0), 2 * (self.padding[0],), 2 * (self.padding[1],), (0, 0))
        padded_input = np.pad(propagated_input, pad_widths, mode="constant")
        self.most_recent_propagated_input = padded_input
        weighted_output = []
        batch_size = self.input_shape[0]
        for b in range(batch_size):
            frame = padded_input[b]
            filters = []
            for f in range(self.num_filters):
                filter_ = self.W[f]
                strides_view_shape = tuple(1 + np.subtract(frame.shape[:-1], filter_.shape[:-1])) + filter_.shape
                a, b, c = frame.strides
                conv_windows = np.lib.stride_tricks.as_strided(frame, strides_view_shape, (b, a, a, b, c))
                filters.append(np.einsum("klm,ijklm->ij", filter_, conv_windows) + self.b[f])
            weighted_output.append(filters)
        activation_output = self.activation.compute(weighted_output)
        return activation_output

    def backward(self, back_propagated_gradients):
        activation_gradients = back_propagated_gradients * self.activation.gradients()
        result = []
        for b in range(self.input_shape[0]):
            xx = self.most_recent_propagated_input[b]
            r = []
            for f in range(self.num_filters):
                ff = self.W[f]
                gg = activation_gradients[b][f]
                view_shape = ff.shape[:-1] + tuple(1 + np.subtract(xx.shape[:-1], ff.shape[:-1])) + (ff.shape[-1],)
                a, b, c = xx.strides
                conv_windows = np.lib.stride_tricks.as_strided(xx, view_shape, (b, a, a, b, c))
                cc = np.einsum("lk,ijklm->jim", gg, conv_windows)
                r.append(cc)
            result.append(r)
        self.dW = np.mean(result, axis=0)
        self.db = np.mean(activation_gradients, axis=(0, 2, 3)).reshape(self.b.shape)
        if not self.is_first_layer:
            layer_grads = np.zeros(self.input_shape)
            for b in range(self.input_shape[0]):
                xx = layer_grads[b]
                for f in range(self.num_filters):
                    ff = self.W[f]
                    gg = activation_gradients[b][f]
                    view_shape = tuple(1 + np.subtract(xx.shape[:-1], ff.shape[:-1])) + ff.shape
                    a, b, c = xx.strides
                    conv_windows = np.lib.stride_tricks.as_strided(xx, view_shape, (b, a, a, b, c))
                    conv_windows += np.tile(ff, (*gg.shape, 1, 1, 1)) * gg.reshape((*gg.shape, 1, 1, 1))
            return layer_grads

    @property
    def parameters_and_gradients(self):
        return [(self.W, self.dW), (self.b, self.db)]
