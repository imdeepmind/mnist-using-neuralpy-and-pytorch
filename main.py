from neuralpy.models import Model
from neuralpy.loss_functions import CrossEntropyLoss
from neuralpy.optimizer import Adam

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

X = X.reshape(len(X), 1, 28, 28)
X_test = X_test.reshape(len(X_test), 1, 28, 28)

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
# Initializing the PyTorch model
p_model = MNISTModel()

# Initializing a NeuralPy model
model = Model()

# Converting the PyTorch model to NeuralPy model
model.set_model(p_model)

# Printing the summary of the model
print(model.summary())

# Compiling the model
model.compile(optimizer=Adam(), loss_function=CrossEntropyLoss(), metrics=["accuracy"])

# Training the model
# Using the fit method
history = model.fit(train_data=(X_train, y_train), test_data=(X_validation, y_validation), epochs=10, batch_size=32)

# Evaluatung the model
ev = model.evaluate(X=X_test, y=y_test, batch_size=32)

print(f"Loss: {ev['loss']} and accuracy: {ev['accuracy']}%")