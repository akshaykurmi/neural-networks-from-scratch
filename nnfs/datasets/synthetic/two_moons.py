import numpy as np
import matplotlib.pyplot as plt


class TwoMoons:
    def __init__(self, n_samples, noise=0.1, random_state=0):
        self.n_samples = n_samples
        self.noise = noise
        self.random_state = np.random.RandomState(random_state)

    def generate(self):
        upper_moon_samples = self.n_samples // 2
        lower_moon_samples = self.n_samples - upper_moon_samples
        upper_moon_x = np.cos(np.linspace(0, np.pi, upper_moon_samples))
        upper_moon_y = np.sin(np.linspace(0, np.pi, upper_moon_samples))
        lower_moon_x = 1 - np.cos(np.linspace(0, np.pi, lower_moon_samples))
        lower_moon_y = 1 - np.sin(np.linspace(0, np.pi, lower_moon_samples)) - 0.5
        X = np.vstack((np.append(upper_moon_x, lower_moon_x), np.append(upper_moon_y, lower_moon_y))).T
        y = np.hstack([np.zeros(upper_moon_samples, dtype=np.intp), np.ones(lower_moon_samples, dtype=np.intp)])
        X += self.random_state.normal(scale=self.noise, size=X.shape)
        return X, y

    def visualize(self):
        X, y = self.generate()
        colors = ['b' if i == 0 else 'r' for i in y]
        plt.scatter(X.T[0], X.T[1], c=colors, marker='^')
        plt.show()
