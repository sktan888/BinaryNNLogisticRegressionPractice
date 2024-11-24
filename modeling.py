# pylint: disable=pointless-statement
#!/usr/bin/env python3

# python modeling.py modeling-cmd N where N is the digit to classify

import myLib.data
import myLib.helper

import numpy as np
import click
from myLib.mylog import log


@click.group()
def cli():
    """run NN modelling"""


@cli.command()
@click.argument("digit", type=int)
def modeling_cmd(digit):

    # injest datasets
    train_set_x, train_set_y, test_set_x, test_set_y = myLib.data.injest(digit)

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
    # index = 6
    # print("Actual = " + str(test_set_y[:, index]))
    # print("Prediction = " + str(logistic_regression_model["Y_prediction_test"][0, index]))
    # assert test_set_y[:, index] == logistic_regression_model["Y_prediction_test"][0, index]

    # save model to an NPY file
    np.save("model_weights.npy", logistic_regression_model["w"])
    np.save("model_bias.npy", np.array([logistic_regression_model["b"]]))

    # save datasets to an NPY file
    np.save("test_set_x.npy", test_set_x)
    np.save("test_set_y.npy", test_set_y)

    click.echo("Cost = " + str(np.squeeze(logistic_regression_model["costs"])))
    log("Cost = " + str(np.squeeze(logistic_regression_model["costs"])))


if __name__ == "__main__":
    cli()
