# pylint: disable=pointless-statement, unused-variable
#!/usr/bin/env python3

# python predictingUnknownImage.py predict_cmd Image where Image is the filename nine.jpg uploaded in assets/images folder

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
@click.argument("fileName")  # the name of the image file
def predict_cmd(fileName):
    """predict a given image"""

    # injest test datasets from the NPY file
    test_set_x = np.load("test_set_x.npy")
    num_px = 28  # test_set_x.shape[1]
    log("num_px " + str(num_px))

    # injest image file my_image.jpg
    # Preprocess the image to fit  the NN algorithm.
    fname = "assets/images/" + fileName
    image = np.array(Image.open(fname).resize((num_px, num_px)))
    log("image.shape " + str(image.shape))

    plt.imshow(image)
    image = image / 255.0
    image = image.reshape((1, num_px * num_px * 3)).T

    # Load trained model from the NPY file
    w = np.load("model_weights.npy")
    b = np.load("model_bias.npy")[
        0
    ]  # convert a Python array with a single element to a scalar

    a = "Actual = " + str(fileName)
    p = "Prediction = " + str(predict(w, b, image))
    click.echo(a)
    click.echo(p)
    log("Example " + str(fileName) + " :: " + a + " : " + p)


if __name__ == "__main__":
    cli()
