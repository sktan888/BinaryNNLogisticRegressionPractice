extern crate mnist;
//use itertools::Itertools;
use mnist::*;
use ndarray::{Array2, Array3}; //, Zip};

/*
fn find_index<T: PartialEq>(arr: &[T], target: &T) -> Option<usize> {
    for (i, x) in arr.iter().enumerate() {
        if x == target {
            return Some(i);
        }
    }
    None
}
*/
fn find_indices_filter<T: PartialEq>(arr: &[T], target: &T) -> Vec<usize> {
    arr.iter().enumerate()
        .filter(|(_, x)| **x == *target)
        .map(|(i, _)| i)
        .collect()
}
fn find_indices_filter2<T: PartialEq>(arr: &[f32], target: &f32) -> Vec<usize> {
    arr.iter().enumerate()
        .filter(|(_, x)| **x == *target)
        .map(|(i, _)| i)
        .collect()
}
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

    let _image_num = 0;

    let train_data = Array3::from_shape_vec((60_000, 28, 28), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.0);

    // Convert the returned Mnist struct to Array2 format
    let train_labels: Array2<f32> = Array2::from_shape_vec((60_000, 1), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f32);

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

    // In handwriting dataset, y is digits 0 to 9 and requires 10 output neurons to classify all 10 digits
    // Since this is single output NN, consider classifying one digit at one time for now
    // train_set_y zeros for non N and ones for N
    // ???
    //    train_set_y_binary = np.zeros((1, train_set_y.size))
    let train_set_y_binary: Array2<i32> = Array2::zeros((60000, 1));
    assert_eq!(train_set_y_binary.shape(), &[60000, 1]);

    let _array = train_set_y_binary;

    //let target_number = &3;
    //indices = find_indices_filter(array, target_number);

    let numbers = [1, 2, 3, 4, 5];
    let target_number = &3;
    let target_number2 = 3 as f32;
    /*
    if let Some(index) = find_index(&numbers, target_number) {
        println!("Found {} at index {}", target_number, index);
    } else {
        println!("{} not found", target_number);
    }
    */

    let index2 = find_indices_filter(&numbers, target_number);
    println!("Found {} at index {:?}", target_number, index2);

    let (my_vector, _)= train_labels.into_raw_vec_and_offset();

    let index3 = find_indices_filter2(&my_vector, &target_number2);
    println!("Found {} at index {:?}", target_number, index3);

    /*
    let indices = Zip::from(&array)
        .zip(Zip::indices(&array))
        .filter_map(|(x, (i, j))| if x != 0 { Some((i, j)) } else { None })
        .collect::<Vec<_>>();
    */
    //println!("Non-zero indices: {:?}", indices);

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

    // Flatten train dataset from array(60000, 28, 28) into (784,60000) and test dataset from (10000,28,28) to (748, 10000)

    assert_eq!(train_data.shape(), &[60000, 28, 28]);
    // Reshape the array into a 2D array with shape (60000, 784)
    let reshaped_data = train_data.to_shape((60000, 784)).unwrap();
    // Transpose the reshaped array to get the desired shape (784, 60000)
    let flattened_train_data = reshaped_data.t();
    println!("Flattened train shape: {:?}", flattened_train_data.shape());

    let reshaped_data = test_data.to_shape((10000, 784)).unwrap();
    // Transpose the reshaped array to get the desired shape (784, 10000)
    let flattened_test_data = reshaped_data.t();
    println!("Flattened test shape: {:?}", flattened_test_data.shape());

    /*
    let dims = flattened_train_data.dim(); // This is of type `ndarray::Dim<[usize; 3]>`
    //let (_row, _col) = dims.into(); // Extract dimensions as tuple
    println!("Flattened train dim: {:?}", dims);

    // Create a 2D array
    let array: Array2<i32> = arr2(&[[1, 2, 3], [4, 5, 6]]);
    println!("Shape of  2D array: {:?}", array.shape());

    let (flattened_vec, _) = array.into_raw_vec_and_offset();
    println!("Shape of flattened_vec: {:?}", flattened_vec.len());

    let _nested_array: [[i32; 3]; 2] = [[1, 2, 3], [4, 5, 6]];

    let vec3d: Vec<Vec<Vec<i32>>> = vec![
        vec![vec![1, 2, 3], vec![4, 5, 6]],
        vec![vec![7, 8, 9], vec![10, 11, 12]],
    ];
    let element = vec3d[1][0][2]; // Accesses the element at (1, 0, 2)

    println!("element: {:?}", element);


    let a: Array3<i32> = Array3::zeros((8, 5, 2));

    println!("Shape of array3d: {:?}", a.shape());

    let _array_dd: Array3<i32> = Array3::zeros((2, 3, 4)); // Create a 2x3x4 array filled with zeros

    let a = arr3(&[
        [
        [ 1,  2,  3],     // -- 2 rows  \_
        [ 4,  5,  6]],    // --         /
        [
        [ 7,  8,  9],     //            \_ 2 submatrices
        [10, 11, 12]
        ]
        ]);  //            /
        //  3 columns ..../.../.../


    let b = arr3(&[
        [
        [ 1,  2,  3],     // -- 2 rows  \_
        [ 4,  5,  6]
        ]
        ]);  //            /
        //  3 columns ..../.../.../

    let c = arr2(&[
        [ 1,  2,  3],
        [ 4,  5,  6]
        ]);


    assert_eq!(a.shape(), &[2, 2, 3]);
    assert_eq!(b.shape(), &[1, 2, 3]);
    assert_eq!(c.shape(), &[2, 3]);
    println!("Shape of array3d: {:?}", a.shape());
    println!("Shape of array2d: {:?}", b.shape());
    println!("Shape of array2d: {:?}", c.shape());

        // Access the dimension information
        /*
        let dim = arr.dim(); // This is of type `ndarray::Dim<[usize; 3]>`
        let (x, y, z) = dim.into_tuple(); // Extract dimensions as tuple
        */
    */
}
