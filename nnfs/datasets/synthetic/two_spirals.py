import numpy as np
import matplotlib.pyplot as plt


class TwoSpirals:
    def __init__(self, n_samples, noise=0.5, random_state=0, winding=700):
        self.n_samples = n_samples
        self.noise = noise
        self.winding = winding
        self.random_state = np.random.RandomState(random_state)

    def generate(self):
        noise = self.random_state.rand(self.n_samples, 1) * self.noise
        samples = self.random_state.rand(self.n_samples, 1)
        samples = np.sqrt(samples) * self.winding * np.pi / 180
        xx = -np.cos(samples) * samples + noise
        yy = np.sin(samples) * samples + noise
        X = np.vstack((np.hstack((xx, yy)), np.hstack((-xx, -yy))))
        y = np.hstack((np.zeros(self.n_samples), np.ones(self.n_samples)))
        return X, y

    def visualize(self):
        X, y = self.generate()
        colors = ['b' if i == 0 else 'r' for i in y]
        plt.scatter(X.T[0], X.T[1], c=colors, marker='^')
        plt.show()
