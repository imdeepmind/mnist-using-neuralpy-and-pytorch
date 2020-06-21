from neuralpy.models import Model

import pandas as pd

from model import Net as MNISTModel

## Loading the data and data preprocessing
# Loading the data
train_data = pd.read_csv("./data/mnist_train.csv")
test_data = pd.read_csv("./data/mnist_test.csv")

# Shuffling and converting the DataFrame to numpy array
train_data = train_data.sample(frac=1)
train_data = train_data.values

# Shuffling and converting the DataFrame to numpy array for test data
test_data = test_data.sample(frac=1)
test_data = test_data.values

# Extracting the X and y
X = train_data[:, 1:] / 255.
y = train_data[:, 0]

# Extracting the X and y
X_test = test_data[:, 1:]
y_test = test_data[:, 0]

del train_data

train_data_size = .8

X_train = X[0:int(len(X)*train_data_size)]
y_train = y[0:int(len(y)*train_data_size)]

X_validation = X[int(len(X)*train_data_size):]
y_validation = y[int(len(y)*train_data_size):]

print("Size of Train data is", len(X_train))
print("Size of Validation data is ", len(X_validation))
print("Size of Test data is ", len(X_test))

# Making the model








p_model = MNISTModel()

model = Model()
model.set_model(p_model)

print(model.summary())