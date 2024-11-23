# pylint: disable=pointless-statement

import myLib.data
import numpy as np

# injest datasets
train_set_x, train_set_y, test_set_x, test_set_y = myLib.data.injest(4)

# EDA


# train model
logistic_regression_model = myLib.helper.model(
    train_set_x,
    train_set_y,
    test_set_x,
    test_set_y,
    num_iterations=100,
    learning_rate=0.005,
    print_cost=True,
)

# evaluate predictions on the test set
index = 6
print("Actual = " + str(test_set_y[:, index]))
print("Prediction = " + str(logistic_regression_model["Y_prediction_test"][0, index]))
assert test_set_y[:, index] == logistic_regression_model["Y_prediction_test"][0, index]

# save model to an NPY file
np.save('model_weights.npy', logistic_regression_model["w"])
np.save('model_bias.npy', np.array([logistic_regression_model["b"]]) )