from neuralpy.models import Model

import pandas as pd

from model import Net as MNISTModel

# Loading the data and data preprocessing
train_data = pd.read_csv("./data/mnist_train.csv", header=None)
test_data = pd.read_csv("./data/mnist_test.csv", header=None)

train_data = train_data.sample(frac=1)
train_data = train_data.values

test_data = test_data.sample(frac=1)
test_data = test_data.values

X = train_data[:, 1:] / 255.
y = train_data[:, 0]

X_test = test_data[:, 1:]
y_test = test_data[:, 0]

del train_data









p_model = MNISTModel()

model = Model()
model.set_model(p_model)

print(model.summary())