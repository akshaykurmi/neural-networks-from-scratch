# Neural Networks From Scratch

A pure Numpy deep learning framework with a modular Keras-like API.

### Usage

The following is a simple example of multi-class classification on the Iris Flower dataset.
Checkout the examples directory for more.

```python
from nnfs.datasets import IrisFlowers
from nnfs.activations import ReLU, Softmax
from nnfs.losses import CategoricalCrossentropy
from nnfs.optimizers import Momentum
from nnfs.initializers import RandomUniform, Zeros
from nnfs.metrics import CategoricalAccuracy
from nnfs.layers import Dense, Dropout
from nnfs.models import FeedForwardNetwork
from nnfs.utils.data import split_data
from nnfs.utils.preprocessing import OneHotEncoder, normalize

dataset = IrisFlowers()
dataset.download_data()
X, y = dataset.load_data()

ohe = OneHotEncoder()
y = ohe.fit_transform(y)
X = X.to_numpy()
X_train, y_train, X_test, y_test = split_data(X, y, ratio=0.8)
X_train, X_test = normalize(X_train, X_test)

model = FeedForwardNetwork()
model.add(Dense(units=64,
                activation=ReLU(),
                weights_initializer=RandomUniform(),
                bias_initializer=Zeros(),
                input_shape=(4,)))
model.add(Dropout(0.3))
model.add(Dense(units=3,
                activation=Softmax(),
                weights_initializer=RandomUniform(),
                bias_initializer=Zeros()))
model.compile(optimizer=Momentum(learning_rate=0.001, nesterov=True, clip_value=0.05),
              loss=CategoricalCrossentropy(),
              metrics=[CategoricalAccuracy()]

model.fit(X_train, y_train, batch_size=32, epochs=1000, verbosity=0)
loss, metrics = model.evaluate(X_test, y_test, batch_size=32)
print("Test Loss :", loss)
print("Test Metrics :", metrics)
``` 

### Components Included

| Category                   | Components                            |
| -------------------------- | ------------------------------------- |
| *Layers*                   | Dense                                 |
|                            | Dropout                               |
|                            | Convolution 2D                        |
|                            | MaxPooling 2D                         |
|                            | Flatten                               |
| *Activations*              | ReLU                                  |
|                            | Sigmoid                               |
|                            | Softmax                               |
| *Optimizers*               | Stochastic Gradient Descent           |
|                            | SGD with Momentum                     |
| *Losses*                   | Mean Squared Error                    |
|                            | Categorical Crossentropy              |
| *Weight Initializers*      | Zeros, Ones, Constant                 |
|                            | Random Normal, Random Uniform         |
|                            | He Normal, He Uniform                 |
|                            | Glorot Normal, Glorot Uniform         |
| *Metrics*                  | Categorical Accuracy                  |
| *Datasets*                 | Synthetic - Moons, Spirals            |
|                            | MNIST Handwritten Digits              |
|                            | Wine Quality                          |
|                            | Iris Flower                           |
