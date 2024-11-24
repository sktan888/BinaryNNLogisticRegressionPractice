# pylint: disable=pointless-statement
#!/usr/bin/env python3

import numpy as np
from myLib.helper import predict
from myLib.mylog import log
import matplotlib.pyplot as plt
from PIL import Image

import click


@click.group()
def cli():
    """run NN Prediction"""


@cli.command()
@click.argument("example") # the name of the image file
def predict_cmd(example):
    """predict a given image"""

    # injest test datasets from the NPY file
    test_set_x = np.load("test_set_x.npy")
    num_px = test_set_x.shape[1]

    # injest image file
    # Preprocess the image to fit  the NN algorithm.
    fname = "images/" + example
    image = np.array(Image.open(fname).resize((num_px, num_px)))
    plt.imshow(image)
    image = image / 255.
    image = image.reshape((1, num_px * num_px * 3)).T


    # Load trained model from the NPY file
    w = np.load("model_weights.npy")
    b = np.load("model_bias.npy")[
        0
    ]  # convert a Python array with a single element to a scalar

    a="Actual = " + str(example)
    p="Prediction = " + str(predict(w, b, image))
    click.echo(a)
    click.echo(p)
    log("Example " + str(example) + " :: "  + a + " : " + p)


if __name__ == "__main__":
    cli()
