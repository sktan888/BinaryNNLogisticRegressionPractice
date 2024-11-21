# BinaryNNLogisticRegressionPractice
Binary classification with Neural Network based on Logistic Regression

Step 1: Configure environment
* create virtual environment: ```virtualenv ENV```
* add ```source ENV/bin/activate``` in .bashrc file
* edit bashrc file ```vim ~/.bashrc``` and goes to the last line: ```shift + g``` 
* create Makefile for make utility : ``` touch Makefile ```
* create Requirements.txt for package install : ``` touch Requirement.txt ```
* find out package version" ```pip freeze | less```
* create library folder: ``` mkdir myLib ```
* rename a file: ```mv oldfilename newfilename```

+++ data.py +++

#import numpy as np
import keras as keras

# Loading data for handwriting
train_data, test_data = keras.datasets.mnist.load_data()

train_set_x = train_data[0]  # x_train = train_data[0]
train_set_y = train_data[1] # y_train = train_data[1]

test_set_x = test_data[0] # x_val = test_data[0]
test_set_y = test_data[1] # y_val = test_data[1]

# train set y and test_set_y are originally row vector (m, 1). Reshape to column vector (1,m)
train_set_y=train_set_y.reshape(1,train_set_y.size)
test_set_y=test_set_y.reshape(1,test_set_y.size)

# Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, ...)
m_train = train_set_x.shape[0] # train_set_y.size
m_test = test_set_x.shape[0]
num_px = train_set_x.shape[1]

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ")")
print ("train_set_x shape: " + str(train_set_x.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

+++ requirement,txt

 ipdb==0.13.13
pytest==8.3.3
pytest-cov==5.0.0
pylint==3.3.1
black==24.10.0
ipython==8.29.0
nbval==0.11.0
pandas==2.2.3
click==8.1.7
keras==3.6.0
