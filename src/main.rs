use crum::matrix;
use crum::complex::Complex;
use crum::matrix::Matrix;

fn main() {


   let mut m_a = matrix![[0.0,5.0,22.0/3.0],
                                       [4.0,2.0,1.0],
                                       [2.0,7.0,9.0]];

   let (u,l,p, _) = m_a.lu(1e-15).unwrap();
   println!("u{}\np{}\nl{}",u,p, l);

}
