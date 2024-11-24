use mnist::{Mnist, MnistBuilder};

fn private_fun() {
    println!("called `my_mod::private_function()`");
}

// Loading data for handwriting
pub fn injest(digit: i32) {

    // private_fun();

    // Load the MNIST dataset
    let mnist = MnistBuilder::default()
    .label_format(mnist::LabelFormat::OneHot)
    .finalize()
    .expect("Error initializing MNist");

    // Get the training and test sets
    let (x_train, y_train, x_test, y_test) = mnist.load_mnist(None).expect("Error loading MNIST");

    // Print the shape of the training data
    println!("Shape of x_train: {:?}", x_train.shape());
    println!("Shape of y_train: {:?}", y_train.shape());
}




