import os
import numpy as np

from . import DATA_DIRECTORY


class DinosaurNames:
    DATA_FILE = os.path.join(DATA_DIRECTORY, "dino_names.txt")

    def load_data(self):
        with open(self.DATA_FILE) as f:
            data = f.readlines()
        data = [x.strip() for x in data]
        return np.array(data).reshape(-1, 1)
