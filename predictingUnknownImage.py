import click
from myLib.mylog import log
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from myLib.helper import predict


@click.command()
@click.argument("file_name")
def predicting(file_name):
    """Predict"""
    # injest test datasets from the NPY file
    # test_set_x = np.load("test_set_x.npy")
    num_px = 28  # test_set_x.shape[1]
    log("num_px " + str(num_px))

    # injest image file my_image.jpg
    # Preprocess the image to fit  the NN algorithm.
    fname = "assets/images/" + file_name
    image = np.array(Image.open(fname).resize((num_px, num_px)))
    image = image[:, :, 0]
    # plt.imshow(image)
    log("image.shape " + str(image.shape))
    click.echo(f"image.shape " + str(image.shape))

    image = image / 255.0
    image = image.reshape((1, num_px * num_px)).T

    log("image.shape " + str(image.shape))
    click.echo(f"image.shape " + str(image.shape))

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
    predicting()
