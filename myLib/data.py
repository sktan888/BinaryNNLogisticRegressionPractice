# pylint: disable=E1101, unused-import
import myLib.helper
import numpy as np
import keras as keras

# Loading data for handwriting
train_data, test_data = keras.datasets.mnist.load_data()

train_set_x = train_data[0]  # x_train = train_data[0]
train_set_y = train_data[1]  # y_train = train_data[1]

test_set_x = test_data[0]  # x_val = test_data[0]
test_set_y = test_data[1]  # y_val = test_data[1]

# train set y and test_set_y are originally row vector (m, 1). Reshape to column vector (1,m)
train_set_y = train_set_y.reshape(1, train_set_y.size)
test_set_y = test_set_y.reshape(1, test_set_y.size)

# Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, ...)
m_train = train_set_x.shape[0]  # train_set_y.size
m_test = test_set_x.shape[0]
num_px = train_set_x.shape[1]

print("Number of training examples: m_train = " + str(m_train))
print("Number of testing examples: m_test = " + str(m_test))
print("Height/Width of each image: num_px = " + str(num_px))
print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ")")
print("train_set_x shape: " + str(train_set_x.shape))
print("train_set_y shape: " + str(train_set_y.shape))
print("test_set_x shape: " + str(test_set_x.shape))
print("test_set_y shape: " + str(test_set_y.shape))

# Reshape the training and test examples
train_set_x_flatten = train_set_x.reshape(train_set_x.shape[0], -1).T
test_set_x_flatten = test_set_x.reshape(test_set_x.shape[0], -1).T


print("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print("train_set_y shape: " + str(train_set_y.shape))
print("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print("test_set_y shape: " + str(test_set_y.shape))

# standardize dataset
train_set_x = train_set_x_flatten / 255.0
test_set_x = test_set_x_flatten / 255.0


logistic_regression_model = myLib.helper.model(
    train_set_x,
    train_set_y,
    test_set_x,
    test_set_y,
    num_iterations=2000,
    learning_rate=0.005,
    print_cost=True,
)
