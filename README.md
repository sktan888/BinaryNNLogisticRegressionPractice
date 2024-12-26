[![Python Install/Lint with Github Actions](https://github.com/sktan888/BinaryNNLogisticRegressionPractice/actions/workflows/main.yml/badge.svg)](https://github.com/sktan888/BinaryNNLogisticRegressionPractice/actions/workflows/main.yml)

# LogisticRegression in Python
Binary classification with Logistic Regression 

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
    - install modules ``` make install ```



## Logistic Regression in python
* Injest in data.py ``` train_data, test_data = keras.datasets.mnist.load_data() ```

    The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits. (https://yann.lecun.com/exdb/mnist/)
    - Consisting of 70,000 28x28 black-and-white images of handwritten digits
    - 10 classes 0 to 9: one class for each  digit
    - 60,000 images in the training dataset, 10,000 images in the validation dataset
    - 7,000 images (6,000 train images and 1,000 test images) for each digit/class
    - Classification of MNIST for 0 to 9, would require 10 output NN to classify all 10 digits
    - Binary classification based on logistic regression simplies to telling if a handwriting is the trained digit
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
    - Logistic regression architecture
        - ![NN](/assets/images/nn.png)
    - Sigmoid 
        - ![Sigmoid](/assets/images/sigmoid.png)
    - Binary classification
        - ![LogisticRegression](/assets/images/lr.webp) 
* Conclusion

* Saving trained model parameters (i.e. w and b) in NPY Files
    - NPY files are a binary file format used to store NumPy arrays efficiently storing large arrays and loading back


## Command Line Interface (CLI)
``` python main.py injest 5 ``` 5 refers to the digit for preprocessing dataset

``` python main.py modeling 5 ``` 5 refers to the digit for modeling

``` python main.py predict-test 55 ``` 55 refers to the example index of test dataset

``` python main.py unseen filename ``` filename refers to the image stored in assets/images/ folder

* Test
```
#test_hello.py
from hello import more_hello, more_goodbye, add
def test_more_hello():
    assert "hi" == more_hello()
```
