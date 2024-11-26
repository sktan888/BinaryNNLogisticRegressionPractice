extern crate mnist;
//extern crate tensorflow;

use mnist::{Mnist, MnistBuilder};
//use mnist::*;
// use ndarray::prelude::*;
use ndarray::{Array2, Array1};
//use tensorflow::{Tensor, Session};

fn private_fun() {
    println!("called `my_mod::private_function()`");
}

// Loading data for handwriting pub fn injest(digit: i32) {
pub fn injest(_digit: i32) {
    private_fun();

    // Load the MNIST dataset
    let mnist = MnistBuilder::new()
        .label_format(mnist::LabelFormat::OneHot)
        .finalize()
        .expect("Error initializing MNIST");

    let (x_train, y_train), (x_test, y_test) = mnist.split_test_train();

    /*
    let mnist = MnistBuilder::new()
        .label_format_digit()
        .finalize();

    let (images, labels) = tensorflow::mnist::read_data_sets("path/to/mnist_data");

    // Get the training and test sets
    // let (x_train, y_train, x_test, y_test) = mnist.load_mnist(None).expect("Error loading MNIST");
    let train_images = Array3::from_shape_vec((60000, 28, 28), mnist.trn_img)
        .expect("Error converting training images to Array3 struct")
        .map(|x| *x as f32 / 255.0);

    println!("Shape of train_images: {:?}", train_images.shape());


    let _mnist = MnistBuilder::default()
    .label_format_digit()
    .finalize();


    let _mnist = MnistBuilder::new()
    .label_format_digit()
    .finalize();


    let mnist = MnistBuilder::default()
        .finalize()
        .expect("Error initializing MNist");

    // Get the training and test sets
    let (x_train, y_train, x_test, y_test) = mnist.load_mnist(None).expect("Error loading MNIST");

    // Print the shape of the training data
    println!("Shape of x_train: {:?}", x_train.shape());
    println!("Shape of y_train: {:?}", y_train.shape());
    */
}
