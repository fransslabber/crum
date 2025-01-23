mod complex;
mod cmatrix;

use complex::Complex; // Import the Complex structs
use cmatrix::Matrix;

fn main() {
   // // Example usage with f64
   // let mut c1 = Complex::new(6.0, 4.0);
   // let c2 = Complex::new(3.0, 5.0);
   
   // // (6+4i)mod(3+5i) = -2+2i
   // println!("c1 {:?} c2 {:?} c1 mod c2 {:?}", c1, c2, c1 % c2);
   // // Addition Assign
   // c1 += c2;
   // println!("c1 += c2 {:?}", c1);
   // // Subtraction Assign
   // c1 -= c2;
   // println!("c1 -= c2 {:?}", c1);
   // // Multiplication Assign
   // c1 *= c2;
   // println!("c1 *= c2 {:?}", c1);
   // // Division Assign
   // c1 /= c2;
   // println!("c1 /= c2 {:?}", c1);

   // // Norm
   // println!("sqr|c1| {:?}", c1.norm());

   // // absolute value/modulus/hypotenuse/magnitude
   // println!("|c1| {:?}", c1.hypot());

   // // Real component
   // println!("Real part c1 {:?}", c1.real());

   // // Imaginary component
   // println!("Imaginery part c1 {:?}", c1.imag());

   // // Complex conjugate
   // println!("Complex conjugate c1 {:?}", c1.conj());

   // // Returns a Complex<T> value from polar coords ( angle in radians )
   // println!("Polar to complex number c1 {:?}", Complex::polar(3.0, Complex::degrees_to_radians(45.0)));


   // /*The projection of a complex number is a mathematical operation that maps the complex number to the Riemann sphere, often used in complex analysis. Specifically:

   // For a complex number z=a+biz=a+bi, the projection is defined as:
   // If zz is finite (∣z∣≠∞∣z∣=∞), the projection is zz itself.
   // If zz is infinite (∣z∣=∞∣z∣=∞), the projection maps zz to a "point at infinity." */
   // println!("Projection c1 {:?} to {:?}", c1, c1.proj());

   // // Returns the phase angle (or angular component) of the complex number x, expressed in radians.
   // println!("Phase angle c1 {:?} to {:?} radians", c1, c1.arg());
 
   

   //Use the matrix! macro to create a Matrix<Complex<f64>>
   // let m_complex_f  = matrix![[Complex::new(0.0, 0.0), Complex::new(6.1, -4.0), Complex::new(3.0, -4.0)],
   //                                                 [Complex::new(6.1, 4.0), Complex::new(1.0, 0.0), Complex::new(2.0, -5.0)],
   //                                                 [Complex::new(3.0, 4.0), Complex::new(2.0, 5.0), Complex::new(2.0, 0.0)]];

   // println!("Complex Real {} Diag {:?}", m_complex_f, m_complex_f.diag());

   // let CHT = Matrix::<Complex<f64>>::householder_transform(vec![Complex::new(6.1, 4.0),
   //             Complex::new(1.0, 0.0), Complex::new(2.0, -5.0),Complex::new(3.0, 4.0), Complex::new(2.0, 5.0),
   //             Complex::new(2.0, 0.0)]);
   
   // let X = matrix![[Complex::new(6.1, 4.0)],
   //                                        [ Complex::new(1.0, 0.0)], 
   //                                        [Complex::new(2.0, -5.0)],
   //                                        [Complex::new(3.0, 4.0)], 
   //                                        [Complex::new(2.0, 5.0)], 
   //                                        [Complex::new(2.0, 0.0)]];

   // println!("CHT {} \nX {} \nCHT*X {}",CHT.clone(),X.clone(), CHT*X );


   // let m_complex_f  = matrix![[Complex::new(7.0, 8.9), Complex::new(6.1, -4.0), Complex::new(3.0, -4.0)],
   //                                                 [Complex::new(6.1, 4.0), Complex::new(1.0, 3.0), Complex::new(2.0, -5.0)],
   //                                                 [Complex::new(3.0, 4.0), Complex::new(2.0, 5.0), Complex::new(2.0, 1.2)],
   //                                                 [Complex::new(6.1, 4.0), Complex::new(1.0, 3.0), Complex::new(2.0, -5.0)],
   //                                                 [Complex::new(7.0, 8.9), Complex::new(6.1, -4.0), Complex::new(3.0, -4.0)],
   //                                                 [Complex::new(6.1, 4.0), Complex::new(1.0, 3.0), Complex::new(2.0, -5.0)],
   //                                                 [Complex::new(3.0, 4.0), Complex::new(2.0, 5.0), Complex::new(2.0, 1.2)],
   //                                                 [Complex::new(6.1, 4.0), Complex::new(1.0, 3.0), Complex::new(2.0, -5.0)],
   //                                                 [Complex::new(7.0, 8.9), Complex::new(6.1, -4.0), Complex::new(3.0, -4.0)]];
   
   //println!( "sub-matrix (2..3,2..3) {}", m_complex_f.clone().sub_matrix(2..=3,2..=3));
   
   //println!( "insert (zero)row, insert (unit)col{}", m_complex_f.clone().insert_row(2, vec![ Complex::new(1.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0) ]));
                                               // .insert_col(u128::max_value(), vec![ Complex::new(1.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0),Complex::new(0.0, 0.0) ]));
   
   //println!( "Augment {}", m_complex_f.augment(2));   

   // let (x,y) = Matrix::<Complex<f64>>::qr_cht(m_complex_f.clone());
   // println!( "Q{}\nR{}\nA{}", x.clone(),y.clone(),x.clone()*y.clone());

   // let m_complex_u64 = Matrix::rnd_matrix(10,3,1.0..=101.0);
   // let (x,y) = Matrix::<Complex<f64>>::qr_cht(m_complex_u64.clone());
   // println!( "Original{}\nQ{}\nR{}\nA{}",m_complex_u64, x.clone(),y.clone(),x.clone()*y.clone());

   // let m_identity_4_4 = Matrix::<Complex<f64>>::identity(4);

   // println!("4x4 Complex Identity {} Check {}", m_identity_4_4, m_identity_4_4.is_identity());

   // let m_complex_i  = matrix![[Complex::new(0, 0), Complex::new(6, -4), Complex::new(3, -4)],
   //                                                 [Complex::new(6, 4), Complex::new(1, 0), Complex::new(2, -5)],
   //                                                 [Complex::new(3, 4), Complex::new(2, 5), Complex::new(2, 0)]];
   
   //println!("Complex Integer {}", m_complex_i);

   
   //println!("Is Hermitian {:?}", m_complex_f.clone().is_hermitian());
   //println!("Is Hermitian {:?}", m_complex_i.clone().is_hermitian());
   //let (q,r) = m_complex_f.qr_decomp_gs(); 

   // println!("QR decomposition \n\nQ:\n\n{}\n\nR:\n\n{}",q,r );

   // println!("QR Recombine \n\nQR:\n\n{}",q*r );

   // println!("{:?} \nIndex(2,1) {:?}", m_complex, m_complex[(2,1)]);

   // m_complex[(2,1)] = Complex::new(7.0, 8.0);

   // println!("Index(2,1) {:?}", m_complex[(2,1)]);

   // println!("Row 2 {:?}", m_complex.row(2));

   // println!("Col 2 {:?}", m_complex.col(2));

   // let m = matrix![
   //    [1.0, 2.0],
   //    [3.0, 4.0]
   //    ];

   // let n = matrix![
   //    [1.0, 2.0, 7.0],
   //    [3.0, 4.0, 8.0]
   //    ];

   // let m_mul = m*n.clone();
   // println!("Result: {:?}", m_mul.data());

   //println!("Complex Matrix Multiplication {:?}", m_complex.clone() * Complex::new(6.0, 4.0) + Complex::new(6.0, 4.0));

   // let o = matrix!([[Complex::new(4.6,2.9),Complex::new(6.6,7.3)]]);  // 1x2
   // let p = matrix!([[Complex::new(4.6,2.9)],[Complex::new(6.6,7.3)]]); // 2x1
   // println!("Sim Vector Dot {:?}", o.clone()*p);

   // println!("Scalar multiplication {:?}", n.mul_scalar(&3.0));

   //let mut m_complex_rep: Matrix<Complex<f64>> = matrix!(3;3; Complex::new(6.0, 4.0); Complex::<f64> );

   // let n = matrix![
   //    [1.0, 2.0, 7.0],
   //    [3.0, 4.0, 8.0]];

   // println!("transpose {:?}", n.trans());

   let m_complex_f64 = matrix![[Complex::new(0.9501,0.0),Complex::new( 0.8913,0.0),Complex::new(0.8214,0.0),Complex::new(0.9218,0.0)],
                                                   [Complex::new(0.2311,0.0),Complex::new(0.7621,0.0),Complex::new(0.4447,0.0),Complex::new(0.7382,0.0)],
                                                   [Complex::new(0.6068,0.0),Complex::new(0.4565,0.0),Complex::new(0.6154,0.0),Complex::new(0.1763,0.0)],
                                                   [Complex::new(0.4860,0.0),Complex::new(0.0185,0.0),Complex::new(0.7919,0.0),Complex::new(0.4057,0.0)]];  

   println!( "Schur Decomposition {:?}",m_complex_f64.schur().sub_matrix(2..=3, 2..=3));
   


}
