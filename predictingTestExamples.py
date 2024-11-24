# pylint: disable=pointless-statement
#!/usr/bin/env python3

import numpy as np
from myLib.helper import predict
from myLib.mylog import log

import click


@click.group()
def cli():
    """run NN Prediction"""


@cli.command()
@click.argument("example", type=int)
def predict_cmd(example):
    """predict a given example of test dataset"""

    # injest test datasets from the NPY file
    test_set_x = np.load("test_set_x.npy")
    test_set_y = np.load("test_set_y.npy")

    # Load trained model from the NPY file
    w = np.load("model_weights.npy")
    b = np.load("model_bias.npy")[
        0
    ]  # convert a Python array with a single element to a scalar
    test_set_x_example = test_set_x[:, example].reshape(test_set_x[:, example].size, 1)
    a="Actual = " + str(test_set_y[:, example])
    p="Prediction = " + str(predict(w, b, test_set_x_example))
    click.echo(a)
    click.echo(p)
    log("Example " + str(example) + " :: "  + a + " : " + p)


if __name__ == "__main__":
    cli()
