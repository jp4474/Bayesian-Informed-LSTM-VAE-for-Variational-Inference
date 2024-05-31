import numpy as np
import pandas as pd
import pickle
import os
import random
from sklearn.model_selection import train_test_split



if __name__ == "__main__":
    filenames = os.listdir('data/processed')
    random.shuffle(filenames)
    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10

    # Split filenames into train, validation, and test
    train_filenames, temp_filenames = train_test_split(filenames, test_size=1 - train_ratio)
    validation_filenames, test_filenames = train_test_split(temp_filenames, test_size=test_ratio/(test_ratio + validation_ratio))

    # Write filenames to separate files
    with open('train_files.txt', 'w') as f:
        for filename in train_filenames:
            f.write("%s\n" % filename)

    with open('validation_files.txt', 'w') as f:
        for filename in validation_filenames:
            f.write("%s\n" % filename)

    with open('test_files.txt', 'w') as f:
        for filename in test_filenames:
            f.write("%s\n" % filename)