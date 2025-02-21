use tensor_macro::tensor;

fn main() {
    // Use the `flatten!` macro
    let (flattened, sizes):(Vec<i32>,Vec<usize>) = tensor!([ [[1, 2,9], [3, 4,10]], [[5, 6,11], [7, 8,12]]  ]);

    println!("Flattened: {:?}", flattened); // Output: [1, 2, 3, 4, 5, 6, 7, 8]
    println!("Sizes: {:?}", sizes); // Output: [2, 2, 2]
}