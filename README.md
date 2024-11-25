[![Python Install/Lint with Github Actions](https://github.com/sktan888/BinaryNNLogisticRegressionPractice/actions/workflows/main.yml/badge.svg)](https://github.com/sktan888/BinaryNNLogisticRegressionPractice/actions/workflows/main.yml)

[![Clippy Rust Lint with Github Actions](https://github.com/sktan888/BinaryNNLogisticRegressionPractice/actions/workflows/rust.yml/badge.svg)](https://github.com/sktan888/BinaryNNLogisticRegressionPractice/actions/workflows/rust.yml)

# BinaryNNLogisticRegressionPractice
Binary classification with Logistic Regression based Neural Network 

## Set up working environment
* create virtual environment: ```virtualenv ENV```
    - remove directory: ``` rm -r hENV```
* add ```source ENV/bin/activate``` in .bashrc file
    edit bashrc file ```vim ~/.bashrc``` and goes to the last line: ```shift + g``` 
* create Makefile for make utility : ``` touch Makefile ```
    - rename a file: ```mv oldfilename newfilename```
* create Requirements.txt for package install : ``` touch Requirement.txt ```
    - find out package version ```pip freeze | less```
    - create library folder: ``` mkdir myLib ```


## Neural Network in python
* Injest in data.py ``` train_data, test_data = keras.datasets.mnist.load_data() ```

    The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits. (https://yann.lecun.com/exdb/mnist/)
    - Consisting of 70,000 28x28 black-and-white images of handwritten digits
    - 10 classes 0 to 9: one class for each  digit
    - 60,000 images in the training dataset, 10,000 images in the validation dataset
    - 7,000 images (6,000 train images and 1,000 test images) for each digit/class
    - Classifying MNIST for digits 0 to 9, would require 10 output neurons to classify all 10 digits
    - In binary classification i.e. single output NN, could classify a given single digit 
    ``` 
    train_set_y_binary = np.zeros((1, train_set_y.size))
    classN = digit  # digit to classify
    index = np.where(
        train_set_y == classN
    )  # index of (elements in train_set_y equals classN)
    train_set_y_binary[0, index[1]] = 1
    ```
* EDA
    - Each image is represented as an array of pixels of shape=(28, 28, 1), dtype=uint8
    - Each pixel is an integer between 0 and 255 
    - Label of the image is the numerical digit
    - Visualization:
        - ![Handwriting](/assets/images/digitHW.png)

* Modelling
    - NN with input layer (as many nodes X features), a hidden layer (one node) and an output layer (one node for Binary output)
        - ![NN](/assets/images/nn.png)
    - Sigmoid
        - ![Sigmoid](/assets/images/sigmoid.png)
    - Logistic Regression
        - ![LogisticRegression](/assets/images/lr.webp) 
* Conclusion

* Saving trained NN parameters (i.e. w and b) in NPY Files
    - NPY files are a binary file format used to store NumPy arrays efficiently storing large arrays and loading back
```
import numpy as np

# Save the array to an NPY file in modelling.py
# save model to an NPY file
np.save('model_weights.npy', logistic_regression_model["w"])
np.save('model_bias.npy', np.array([logistic_regression_model["b"]]) )

# save datasets to an NPY file
np.save('test_set_x.npy', test_set_x)
np.save('test_set_y.npy', test_set_y)

# Load the array from the NPY file in predicting.py
# injest test datasets from the NPY file
test_set_x = np.load('test_set_x.npy')
test_set_y = np.load('test_set_y.npy')

# Load trained model from the NPY file
w = np.load('model_weights.npy')
b = np.load('model_bias.npy')[0] # convert a Python array with a single element to a scalar
```

* Command Line Interface (CLI)
    - ``` python predicting.py predict-cmd 55 ``` 55 refers to example in test dataset
```
## predicting.py

#!/usr/bin/env python3

"""The env command allows users to display the current environment or run a specified command in a changed environment."""

import click

@click.group()
def cli():
    """run NN Prediction"""
@cli.command()
@click.argument("example", type=int)
def predict_cmd(example):
    click.echo(a)
    click.echo(p)
    log("Example " + str(example) + " :: "  + a + " : " + p)

if __name__ == "__main__":
cli()
```

* Test
```
#test_hello.py

from hello import more_hello, more_goodbye, add


def test_more_hello():
    assert "hi" == more_hello()


def test_more_goodbye():
    assert "bye" == more_goodbye()


def test_add():
    assert 10 == add(5, 5)

```
## Neural Network in RUST