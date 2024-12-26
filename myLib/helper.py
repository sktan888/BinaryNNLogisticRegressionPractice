# pylint: disable=unused-variable

# FUNCTION: sigmoid
# Mapping the predicted values to probability between 0 and 1 that an instance belongs to a given class or not

import numpy as np
import copy
from myLib.mylog import log

def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z), a probability between 0 and 1
    """

    s = 1 / (1 + np.exp(-z))

    return s


# FUNCTION: initialize_with_zeros
# Produce w, column vector and b, a scalar for NN with input layer (as many nodes X features), a hidden layer (one node) and an output layer (one node for Binary output))


def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias) of type float
    """

    w = np.zeros((dim, 1), dtype=float)
    b = 0.0

    return w, b


# FUNCTION: propagate
# Forward propagation from X to Cost computes cost
# Backward propagation from Cost to X computes gradient dw and db


def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    grads -- dictionary containing the gradients of the weights and bias
            (dw -- gradient of the loss with respect to w, thus same shape as w)
            (db -- gradient of the loss with respect to b, thus same shape as b)
    cost -- negative log-likelihood cost for logistic regression

    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """

    m = X.shape[1]

    # FORWARD PROPAGATION (FROM X TO COST)
    # compute activation
    # compute cost by using np.dot to perform multiplication.
    # And don't use loops for the sum.

    A = sigmoid(np.dot(np.transpose(w), X) + b)
    cost = -(1 / m) * np.sum((Y * np.log(A) + (1 - Y) * np.log(1 - A)))


    #print(f"Y * np.log(A): {Y * np.log(A)}")
    #print(f"A: {A}")
    #print(f"cost: {cost}")

    # BACKWARD PROPAGATION (TO FIND GRAD)

    dw = (1 / m) * np.dot(X, np.transpose(A - Y))
    db = (1 / m) * np.sum(A - Y)

    cost = np.squeeze(np.array(cost))

    grads = {"dw": dw, "db": db}

    return grads, cost


# FUNCTION: optimize
# Optimize w and b by iterating over multiple times forward and backward propagation over the same training dataset
# Update w and b in each iteration through adjustment using the learning rate


def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """

    log("optimize starts")

    w = copy.deepcopy(w)
    b = copy.deepcopy(b)

    costs = []

    for i in range(num_iterations):
        # Cost and gradient calculation
        # grads, cost = ...
        grads, cost = propagate(w, b, X, Y)

        # YOUR CODE ENDS HERE

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # update rule (≈ 2 lines of code)
        w -= learning_rate * dw
        b -= learning_rate * db

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

            # Print the cost every 100 training iterations
            if print_cost:
                print("Cost after iteration %i: %f %f" % (i, cost, b))

    params = {"w": w, "b": b}

    grads = {"dw": dw, "db": db}

    return params, grads, costs


# FUNCTION: predict
# Compute the probability of unknown example X from the trained w and b


def predict(w, b, X):
    """
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    """

    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(np.transpose(w), X) + b)

    # Using loop
    for i in range(A.shape[1]):

        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0

    # Using no loop for better efficieny
    # Y_prediction[A > 0.5] = 1

    return Y_prediction


# FUNCTION: model
# Produce a trained model from training set


def model(
    X_train,
    Y_train,
    X_test,
    Y_test,
    num_iterations=2000,
    learning_rate=0.5,
    print_cost=False,
):
    """
    Builds the logistic regression model by calling the function you've implemented previously

    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to True to print the cost every 100 iterations

    Returns:
    d -- dictionary containing information about the model.
    """

    # initialize parameters with zeros
    # w, b = ...

    # Gradient descent
    # params, grads, costs = ...

    # Retrieve parameters w and b from dictionary "params"
    # w = ...
    # b = ...

    # Predict test/train set examples
    # Y_prediction_test = ...
    # Y_prediction_train = ...

    w, b = initialize_with_zeros(X_train.shape[0])
    params, grads, costs = optimize(
        w, b, X_train, Y_train, num_iterations, learning_rate, print_cost
    )
    w = params["w"]
    b = params["b"]
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print train/test Errors
    if print_cost:
        print(
            "train accuracy: {} %".format(
                100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100
            )
        )
        print(
            "test accuracy: {} %".format(
                100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100
            )
        )

    # 3 Dec 24
    # index is tuple of arrays 
    index = np.where(
        Y_prediction_test == 1
    )  # index of (elements in Y_prediction_test equals 1)
    _, col_index = index 
    # log("Found given digit %i times out of total %i in Y_prediction_test = " % col_index.size, %Y_prediction_test.size + str(np.squeeze(col_index))) #
    # log(f"Found given digit {col_index.size} times out of total {Y_prediction_test.size} in Y_prediction_test = {np.squeeze(col_index)}")
    log(f"Found given digit {col_index.size} times out of total {Y_prediction_test.size} in Y_prediction_test")

    index = np.where(
        Y_test == 1
    )  # index of (elements in X_test equals 1)
    _, col_index = index 
    #log("Found given digit %i times  in Y_test = "  % col_index.size + str(np.squeeze(col_index))) #
    log(f"Found given digit {col_index.size} times out of total {Y_test.size} in Y_test")

    index = np.where(
        Y_prediction_train == 1
    )  # index of (elements in Y_prediction_train equals 1)
    _, col_index = index 
    #log("Found given digit %i times  in Y_prediction_train = "  % col_index.size + str(np.squeeze(col_index))) #
    log(f"Found given digit {col_index.size} times out of total {Y_prediction_train.size} in Y_prediction_train ")

    index = np.where(
        Y_train == 1
    )  # index of (elements in Y_train equals 1)
    _, col_index = index 
    #log("Found given digit %i times  in Y_train = "   % col_index.size + str(np.squeeze(col_index))) #
    log(f"Found given digit {col_index.size} times out of total {Y_train.size} in Y_train ")

    d = {
        "costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediction_train": Y_prediction_train,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations,
    }

    return d
