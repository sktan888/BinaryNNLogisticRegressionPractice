# pylint: disable=pointless-statement

import myLib.data

# injest
train_set_x, train_set_y, test_set_x, test_set_y = myLib.data.injest(4)

# EDA


# model
logistic_regression_model = myLib.helper.model(
    train_set_x,
    train_set_y,
    test_set_x,
    test_set_y,
    num_iterations=100,
    learning_rate=0.005,
    print_cost=True,
)

# evaluate
# Predictions on the test set
index = 6
print("Actual = " + str(test_set_y[:, index]))
print("Prediction = " + str(logistic_regression_model["Y_prediction_test"][0, index]))
# assert test_set_y[:, index] == logistic_regression_model["Y_prediction_test"][0, index]

# save model
