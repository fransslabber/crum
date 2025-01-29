// use crum::matrix;
// use crum::complex::Complex;
// use crum::matrix::Matrix;

fn main() {
   use crum::complex::Complex;
   use crum::matrix::Matrix;

   let mut m_id = Matrix::<Complex<f64>>::identity(3);
   println!("{}",m_id.swap_rows(1,3));

}
