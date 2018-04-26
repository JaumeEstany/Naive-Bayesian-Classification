import numpy as np

def load_dataset(path):

    data = np.genfromtxt(path, skip_header=1, delimiter=';', comments="!!!!!!!!!!!!!!!!!!", dtype=str)

    tweets = data[:, 1]
    classif = data[:, 3].astype(int)

    return tweets, classif