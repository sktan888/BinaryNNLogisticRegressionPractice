# pylint: disable=pointless-statement, unnecessary-pass
#!/usr/bin/env python3
# python main.py

import myLib.data
import myLib.helper
from PIL import Image
import numpy as np
import click
from myLib.mylog import log
from myLib.helper import predict


@click.group()
def cli():
    """run NN modelling and prediction"""
    pass


@cli.command()
@click.argument("digit", type=int)
def modeling(digit):
    """train NN model weights and bias to classify a given handwriting digit supplied in the argument"""

    # injest datasets
    train_set_x, train_set_y, test_set_x, test_set_y = myLib.data.injest(digit)

    # EDA

    # train model
    logistic_regression_model = myLib.helper.model(
        train_set_x,
        train_set_y,
        test_set_x,
        test_set_y,
        num_iterations=10,
        learning_rate=0.005,
        print_cost=True,
    )

    # save model to an NPY file
    np.save("model_weights.npy", logistic_regression_model["w"])
    np.save("model_bias.npy", np.array([logistic_regression_model["b"]]))

    # save datasets to an NPY file
    np.save("test_set_x.npy", test_set_x)
    np.save("test_set_y.npy", test_set_y)

    click.echo("Cost = " + str(np.squeeze(logistic_regression_model["costs"])))
    log("Cost = " + str(np.squeeze(logistic_regression_model["costs"])))

    # 3 Dec 24
    # index is tuple of arrays 
    index = np.where(
        logistic_regression_model["Y_prediction_test"] == 1
    )  # index of (elements in Y_prediction_test equals 1)
    log("b = " + str(np.squeeze(logistic_regression_model["b"])))
    _, col_index = index 

    # log("Predict given digit in Y_prediction_test = " + str(np.squeeze(col_index))) #


@cli.command()
@click.argument("example", type=int)
def predict_test(example):
    """predict a example of test dataset referenced by index supplied in the argument"""

    # injest test datasets from the NPY file
    test_set_x = np.load("test_set_x.npy")
    test_set_y = np.load("test_set_y.npy")

    # Load trained model from the NPY file
    w = np.load("model_weights.npy")
    b = np.load("model_bias.npy")[
        0
    ]  # convert a Python array with a single element to a scalar
    test_set_x_example = test_set_x[:, example].reshape(test_set_x[:, example].size, 1)
    a = "Actual = " + str(test_set_y[:, example])
    p = "Prediction = " + str(predict(w, b, test_set_x_example))
    click.echo(a)
    click.echo(p)
    log("Example " + str(example) + " :: " + a + " : " + p)


@cli.command()
@click.argument("file_name")
def predict_unseen(file_name):
    """Predict unseen example from an image file supplied by file name in the argument"""
    # injest test datasets from the NPY file
    # test_set_x = np.load("test_set_x.npy")
    num_px = 28  # test_set_x.shape[1]
    log("num_px " + str(num_px))

    # injest image file my_image.jpg
    # Preprocess the image to fit  the NN algorithm.
    fname = "assets/images/" + file_name
    image = np.array(Image.open(fname).resize((num_px, num_px)))

    imageori = np.array(Image.open(fname))
    log("imageori shape: " + str(imageori.shape))

    image = image[:, :, 0]
    # plt.imshow(image)
    log("image.shape " + str(image.shape))
    click.echo("image.shape " + str(image.shape))

    image = image / 255.0
    image = image.reshape((1, num_px * num_px)).T

    log("image.shape " + str(image.shape))
    click.echo("image.shape " + str(image.shape))

    # Load trained model from the NPY file
    w = np.load("model_weights.npy")
    b = np.load("model_bias.npy")[
        0
    ]  # convert a Python array with a single element to a scalar

    a = "Actual = " + str(file_name)
    p = "Prediction = " + str(predict(w, b, image))
    click.echo(a)
    click.echo(p)
    log("Example " + str(file_name) + " :: " + a + " : " + p)


if __name__ == "__main__":
    cli()
