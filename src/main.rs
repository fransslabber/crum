mod complex;
mod cmatrix;

use complex::{Complex, IntegerComplex, FloatComplex}; // Import the Complex structs
use cmatrix::Matrix;

fn main() {
   // Example usage with i32
   let c1 = FloatComplex(Complex::new(6.0, 4.0));
   let c2 = FloatComplex(Complex::new(3.0, 5.0));
   // (6+4i)mod(3+5i) = -2+2i
   println!("c1 {:?} c2 {:?} c1 mod c2 {:?}", c1, c2, c1 % c2);
   
   let c3 = IntegerComplex(Complex::new(6, 4));
   let c4 = IntegerComplex(Complex::new(3, 5));
   println!("c3 {:?} c4 {:?} c3 mod c4 {:?}", c3, c4, c3 % c4);

   let m: Matrix<i32> = matrix!([
         [1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]
   ]);

  println!("{:?}", m);

  // Use the matrix! macro to create a Matrix<f64>
  let m_float: Matrix<f64> = matrix!([
         [1.1, 2.2],
         [3.3, 4.4]
   ]);

  println!("{:?}", m_float);
}
