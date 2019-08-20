from tqdm import tqdm
import numpy as np

from ..utils.common import window
from ..utils.data import generate_batches


class FeedForwardNetwork:
    def __init__(self):
        self.optimizer = None
        self.loss = None
        self.metrics = []
        self.layers = []

    def _forward_propagation(self, X, training):
        layer_output = X
        for layer in self.layers:
            layer_output = layer.forward(layer_output, training=training)
        return layer_output

    def _backward_propagation(self, y_pred, y_true):
        previous_layer_gradients = self.loss.gradients(y_pred, y_true)
        for layer in reversed(self.layers):
            previous_layer_gradients = layer.backward(previous_layer_gradients)

    def add(self, layer):
        self.layers.append(layer)
        return self

    def compile(self, optimizer, loss, metrics):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.layers[0].make_first_layer()
        for layer, next_layer in window(self.layers, n=2):
            next_layer.connect_with(layer)

    def fit(self, X, y, batch_size, epochs):
        bar_format = 'Epoch: {n_fmt}/{total_fmt} |{bar}| ETA: {remaining} |{rate_fmt}{postfix}'
        with tqdm(total=epochs, unit="epoch", bar_format=bar_format) as progress_bar:
            for epoch_num in range(epochs):
                predictions = []
                for X_batch, y_batch in generate_batches(batch_size, X, y):
                    y_pred = self._forward_propagation(X_batch, training=True)
                    self._backward_propagation(y_pred, y_batch)
                    self.optimizer.update_parameters(self.layers)
                    predictions.extend(y_pred)
                predictions = np.array(predictions)
                loss = {str(self.loss): self.loss.compute(predictions, y)}
                metrics = {str(metric): metric.compute(predictions, y) for metric in self.metrics}
                progress_bar.update(1)
                progress_bar.set_postfix({**loss, **metrics})

    def predict(self, X, batch_size):
        predictions = []
        for X_batch in generate_batches(batch_size, X):
            y_pred = self._forward_propagation(X_batch, training=False)
            predictions.extend(y_pred)
        return np.array(predictions)

    def evaluate(self, X, y, batch_size):
        y_pred = self.predict(X, batch_size)
        loss = self.loss.compute(y_pred, y)
        metrics = {str(metric): metric.compute(y_pred, y) for metric in self.metrics}
        return loss, metrics
