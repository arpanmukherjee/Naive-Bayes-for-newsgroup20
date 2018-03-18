import numpy as np
from random import shuffle
from test_model import test_naive
from train_model import train_naive
from sklearn.model_selection import train_test_split

data_set = [line.rstrip('\n') for line in open('groundTruth.txt')]
threshold = int(raw_input("Enter threshold:"))
path = str(threshold*10)+'_'+str((10-threshold)*10)


def divide(data):
    X = []
    Y = []
    for sample in data:
        X.append(int(sample.split()[0]))
        Y.append(int(sample.split()[1]))
    return X, Y


data_set = np.array(data_set)
train_X = []
train_Y = []
test_X = []
test_Y = []
i = 0
for c in range(5):
    temp_data = []
    while i < len(data_set) and int(data_set[i].split()[1]) == c:
        temp_data.append(data_set[i])
        i += 1
    shuffle(temp_data)
    train, test = train_test_split(temp_data, train_size=float(threshold)/10)

    x, y = divide(train)
    train_X += x
    train_Y += y
    x, y = divide(test)
    test_X += x
    test_Y += y

train_naive(train_X, train_Y, path)
test_naive(test_X, test_Y, path)
