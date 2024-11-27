extern crate mnist;

//use ndarray::{Array2, Array1};
use mnist::*;
//use ndarray::prelude::*;

use ndarray::{Array2, arr2};
use ndarray::Array3;

// Loading data for handwriting pub fn injest(digit: i32) {
pub fn injest(_digit: i32) {
    // The default path to the MNIST data files is /data of top layer crate

    let m_train = 60_000;
    let m_test = 10_000;
    let _num_px = 28;

    // Download and Load the MNIST dataset
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(m_train)
        .test_set_length(m_test)
        .finalize();
    //.validation_set_length(m_test)
    let _image_num = 0;
    // Can use an Array2 or Array3 here (Array3 for visualization)
    let train_data = Array3::from_shape_vec((60_000, 28, 28), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.0);

    //println!("{:#.1?}\n", train_data.slice(s![image_num, .., ..]));

    // Convert the returned Mnist struct to Array2 format
    let train_labels: Array2<f32> = Array2::from_shape_vec((60_000, 1), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f32);
    /*
    println!(
        "The first digit is a {:?}",
        train_labels.slice(s![image_num, ..])
    );
    */
    let test_data = Array3::from_shape_vec((10_000, 28, 28), tst_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.);

    let test_labels: Array2<f32> = Array2::from_shape_vec((10_000, 1), tst_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as f32);

    // Print the shape of the training data
    println!("Shape of train_data: {:?}", train_data.shape());
    println!("Shape of test_data: {:?}", test_data.shape());

    // # train set y and test_set_y are originally row vector (m, 1) array. Reshape to column vector (1,m) array
    println!("Shape of train_labels: {:?}", train_labels.shape());
    println!("Shape of test_labels: {:?}", test_labels.shape());
    println!("Element of train_labels: {:?}", train_labels[(0, 0)]);

    // Reshape the ArrayView1 into a 2D array
    let _shape = (1, m_train); // Reshape into a (1,m) matrix
    let train_labels_colvector = train_labels.into_shape_with_order((1, 60_000)).unwrap();
    println!(
        "Shape of train_labels_colvector: {:?}",
        train_labels_colvector.shape()
    );

    let test_labels_colvector = test_labels.into_shape_with_order((1, 10_000)).unwrap();
    println!(
        "Shape of test_labels_colvector: {:?}",
        test_labels_colvector.shape()
    );

    // Flatten array(60_000, 28, 28) into (:,60000)


    // Create a 2D array
    let array: Array2<i32> = arr2(&[[1, 2, 3], [4, 5, 6]]);
    println!("Shape of  2D array: {:?}", array.shape());
    
    let (flattened_vec, _) = array.into_raw_vec_and_offset();
    println!("Shape of flattened_vec: {:?}", flattened_vec.len());


    let vec3d: Vec<Vec<Vec<i32>>> = vec![
    vec![vec![1, 2, 3], vec![4, 5, 6]],
    vec![vec![7, 8, 9], vec![10, 11, 12]],
    ];
    let element = vec3d[1][0][2]; // Accesses the element at (1, 0, 2)

    println!("element: {:?}", element);

    // let ndarray_3d: Array3<i32> = Array3::from(vec3d);

    // println!("Shape of array3d: {:?}", ndarray_3d.shape());

    let a: Array3<i32> = Array3::zeros((8, 5, 2));
    
    println!("Shape of array3d: {:?}", a.shape());

    let data = vec![
        vec![vec![1, 2, 3], vec![4, 5, 6]],
        vec![vec![7, 8, 9], vec![10, 11, 12]],
    ];

    let array3d = Array3::from(data);

    println!("{:?}", array3d);
    
    // Access the dimension information
    /*
    let dim = arr.dim(); // This is of type `ndarray::Dim<[usize; 3]>`
    let (x, y, z) = dim.into_tuple(); // Extract dimensions as tuple
    */
}
