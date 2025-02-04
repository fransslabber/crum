use std::ops::Mul;

use crum::matrix;
// use crum::complex::Complex;
use crum::matrix::Matrix;

fn main() {


   // let m_a = matrix![[0.0,5.0,22.0/3.0],
   //                        [4.0,2.0,1.0],
   //                        [2.0,7.0,9.0]];

   //let (l,u,p, swaps) = m_a.lu(1e-15).unwrap();
 
   //println!("swaps {}\nu{}\np{}\nl{}\nPA {}\nLU {}",swaps, u.clone(),p, l.clone(),p.clone() * m_a.clone() ,l.clone()* u.clone());

   // let b = vec![3.0,5.0,7.0];
   // let x = m_a.linear_solve_lu(&b,1e-15).unwrap();
   // assert!( Matrix::<f64>::round_to_decimal_places(x[0],4) == -0.5000 && Matrix::<f64>::round_to_decimal_places(x[1],4) == 5.0000 && Matrix::<f64>::round_to_decimal_places(x[2],4) == -3.0000 );


   let m_b = matrix![[5.0,     2.0,     5.0,     9.0,     5.0,     8.0,     3.0,     6.0,    10.0,     7.0],
                                       [1.0,    10.0,    10.0,     7.0,     1.0,     4.0,     4.0,     3.0,     8.0,     4.0],
                                       [3.0,     1.0,     2.0,     4.0,    10.0,     3.0,     9.0,     8.0,     5.0,     9.0],
                                    [10.0,     8.0,     3.0,     6.0,    10.0,     5.0,     1.0,     2.0,     5.0,     6.0],
                                       [2.0,     9.0,     2.0,     5.0,     5.0,     1.0,     1.0,     7.0,     5.0,     4.0],
                                       [9.0,     9.0,     2.0,     1.0,     5.0,     2.0,     2.0,     2.0,     4.0,    10.0],
                                       [6.0,     1.0,     9.0,     3.0,     4.0,    10.0,     7.0,     4.0,     6.0,     9.0],
                                    [10.0,     4.0,     6.0,     2.0,    10.0,    10.0,     8.0,     7.0,     6.0,     6.0],
                                       [1.0,     3.0,     6.0,     2.0,     4.0,     6.0,     7.0,     8.0,     9.0,     7.0],
                                    [ 5.0,     9.0,     2.0,     3.0,     2.0,     1.0,     5.0,     1.0,     8.0,     6.0]];
   let x = vec![3.0,
   4.0,
   5.0,
   3.0,
   9.0,
   2.0,
   3.0,
   2.0,
   3.0,
   5.0];
   
   let m_x = matrix![[3.0],
   [4.0],
   [5.0],
   [3.0],
   [9.0],
   [2.0],
   [3.0],
   [2.0],
   [3.0],
   [5.0]];


   let b = m_b.clone().mul(m_x);
   let x_computed = m_b.linear_solve_lu(&b.data(),1e-15).unwrap();
   println!("{:?} {:?}",x,x_computed);

}

//
// -0.5000
// 5.0000
// -3.0000
//