# pylint: disable=pointless-statement
#!/usr/bin/env python3

import myLib.data
import myLib.helper
import numpy as np

from myLib.helper import predict
import click

# injest test datasets from the NPY file
test_set_x = np.load("test_set_x.npy")
test_set_y = np.load("test_set_y.npy")

# Load trained model from the NPY file
w = np.load("model_weights.npy")
b = np.load("model_bias.npy")[
    0
]  # convert a Python array with a single element to a scalar

# predict the test set
index = 2
test_set_x_index = test_set_x[:, index].reshape(test_set_x[:, index].size, 1)
print("Actual = " + str(test_set_y[:, index]))
print("Prediction = " + str(myLib.helper.predict(w, b, test_set_x_index)))


@click.group()
def cli():
    """run NN Prediction"""


@cli.command("predict")
@click.argument("index", type=int)
def predict_cmd(example):
    """predict a given example of test dataset"""
    test_set_x_example = test_set_x[:, example].reshape(test_set_x[:, example].size, 1)
    click.echo(predict(w, b, test_set_x_example))
