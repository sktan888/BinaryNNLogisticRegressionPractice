extern crate mnist;
//extern crate tensorflow;

// use ndarray::prelude::*;
//use ndarray::{Array2, Array1};
//use tensorflow::{Tensor, Session};

use mnist::{Mnist, MnistBuilder};
use std::fs::File;
use std::io::Read;



// Loading data for handwriting pub fn injest(digit: i32) {

pub fn injest(_digit: i32) -> Result<(), Box<dyn std::error::Error>> {
    // Specify the path to the MNIST data files
    let train_images_path = "assets/data/mnist-hw/train-images-idx3-ubyte";
    let train_labels_path = "assets/data/mnist-hw/train-labels-idx1-ubyte";
    let test_images_path = "assets/data/mnist-hw/t10k-images-idx3-ubyte";
    let test_labels_path = "assets/data/mnist-hw/t10k-labels-idx1-ubyte";

    // Load the MNIST dataset from the specified paths
    let mnist = MnistBuilder::new()
        .training_set((
            File::open(train_images_path)?,
            File::open(train_labels_path)?,
        ))
        .test_set((
            File::open(test_images_path)?,
            File::open(test_labels_path)?,
        ))
        .finalize()
        .expect("Error initializing MNIST");

    // ... rest of the code as before
}
