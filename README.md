[![Python application test wit Github Actions](https://github.com/sktan888/BinaryNNLogisticRegressionPractice/actions/workflows/main.yml/badge.svg)](https://github.com/sktan888/BinaryNNLogisticRegressionPractice/actions/workflows/main.yml)

# BinaryNNLogisticRegressionPractice
Binary classification with Logistic Regression based Neural Network 

## Set up working environment
* create virtual environment: ```virtualenv ENV```
    - remove directory: ``` rm -f hENV```
* add ```source ENV/bin/activate``` in .bashrc file
    edit bashrc file ```vim ~/.bashrc``` and goes to the last line: ```shift + g``` 
* create Makefile for make utility : ``` touch Makefile ```
    - rename a file: ```mv oldfilename newfilename```
* create Requirements.txt for package install : ``` touch Requirement.txt ```
    - find out package version ```pip freeze | less```
    - create library folder: ``` mkdir myLib ```


## Train Neural Network model
* Injest

    The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits. (https://yann.lecun.com/exdb/mnist/)
    - Consisting of 70,000 28x28 black-and-white images of handwritten digits
    - 60,000 images in the training dataset
    - 10,000 images in the validation dataset
    - 10 classes: one class per digit
    - Per digit/class: 7,000 images (6,000 train images and 1,000 test images)

* EDA
    - Each image is represented as an array of pixels of shape=(28, 28, 1), dtype=uint8
    - Each pixel is an integer between 0 and 255 
    - Label of the image is the numerical digit
    - Visualization:
        - ![Handwriting](/assets/images/hw.png)

* Modelling
    - NN with input layer (as many nodes X features), a hidden layer (one node) and an output layer (one node for Binary output)
        - ![NN](/assets/images/nn.png)
    - Sigmoid
        - ![Sigmoid](/assets/images/sigmoid.png)
* Conclusion




