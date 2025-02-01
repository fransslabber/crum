use crum::matrix;
use crum::complex::Complex;
use crum::matrix::Matrix;

fn main() {


   let m_a = matrix![[4.0,2.0,1.0],[2.0,7.0,9.0],[0.0,5.0,22.0/3.0]];

   let (l,u,p, swaps) = m_a.lu(1e-15).unwrap();
   //println!("{}",u.determinant_lu(swaps));
   // let lhs = p.clone() * m_a.clone();
   // let rhs = l* u;
   // assert!(lhs.data().iter().zip(rhs.data().iter()).all(|(x,y)| x == y ));

   println!("swaps {}\nu{}\np{}\nl{}\nPA {}\nLU {}",swaps, u.clone(),p, l.clone(),p.clone() * m_a.clone() ,l* u);

}
