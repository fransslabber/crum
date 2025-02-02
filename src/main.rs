use crum::matrix;
use crum::complex::Complex;
use crum::matrix::Matrix;

fn main() {


   let m_a = matrix![[0.0,5.0,22.0/3.0],
                          [4.0,2.0,1.0],
                          [2.0,7.0,9.0]];

   let (l,u,p, swaps) = m_a.lu(1e-15).unwrap();
 
   println!("swaps {}\nu{}\np{}\nl{}\nPA {}\nLU {}",swaps, u.clone(),p, l.clone(),p.clone() * m_a.clone() ,l.clone()* u.clone());

   let b = vec![3.0,5.0,7.0];
   let x = Matrix::<f64>::linear_solve_lu(&l, u, p, &b);
   println!("{:?}",x);
}
