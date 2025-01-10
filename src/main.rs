mod complex;
mod cmatrix;

use complex::Complex; // Import the Complex struct
use cmatrix::Matrix;

fn main() {
   // Example usage with i32
   let mut c1 = Complex::new(6, 4 );
   let c2 = Complex::new(3, 5 );
   // (6+4i)mod(3+5i) = -2+2i
   println!("c1 {:?} c2 {:?} c1 mod c2 {:?}", c1, c2, c1 % c2);
   
   // let c3 = c1 * c2;
   // c1 *= c2;
   // let data = vec![Complex::new(1.0,9.0),Complex::new(2.0,10.0),Complex::new(3.0,11.0),Complex::new(4.0,12.0)];
   // let m1 = Matrix::new(2, 2, data);
   // let sldata = vec![Complex::new(5.0,9.0),Complex::new(6.0,10.0),Complex::new(7.0,11.0),Complex::new(8.0,12.0)];
   // let m2 = Matrix::new_with_slice(2, 2, &sldata);
   
   // println!("m1: {:?} m2: {:?} Added {:?}", m1, m2, m1.clone() + m2.clone());

}
