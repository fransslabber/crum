use crum::matrix;
use crum::complex::Complex;
use crum::matrix::Matrix;

fn main() {


   let mut m_id = matrix![[0.0,5.0,22.0/3.0],
                                       [4.0,2.0,1.0],
                                       [2.0,7.0,9.0]];
   let (a,p) = m_id.lu(1e-12);
   println!("a{}\np{}",a,p);

}
