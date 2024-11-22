// A module named `my_mod`

fn private_fun() {
    println!("called `my_mod::private_function()`");
}

// Use the `pub` modifier to override default visibility.
pub fn fun() {
    println!("called `my_mod::function()`");
    private_fun();
}