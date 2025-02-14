use crum::tensor::{contract, Tensor};
use crum::tensor;
use std::ops::RangeInclusive;
use ndarray::{Array4, Array6, Axis};

fn main() {

   let a = tensor!([
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], 
      [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]] 
    ]); // Shape: (2,2,3)
    println!("a {}",a);

    // Define a 3D tensor B of shape (3,4,2)
    let b = tensor!([
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],  // j = 0
        [[9.0, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]],  // j = 1
        [[17.0, 18.0], [19.0, 20.0], [21.0, 22.0], [23.0, 24.0]]  // j = 2
    ]); // Shape: (3,4,2)
   println!("b {}",b);
   println!("{}",contract(1, &a, 2, &b));

   // let b = Tensor::arange(vec![3, 4, 5,6, 2], 1.0);
   // //println!("b {}",b);
   // println!("{}",contract(0, &a, 4, &b));

}
