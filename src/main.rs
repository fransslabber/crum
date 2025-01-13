mod complex;
mod cmatrix;

use complex::Complex; // Import the Complex structs
use cmatrix::Matrix;

fn main() {
   // Example usage with f64
   let mut c1 = Complex::new(6.0, 4.0);
   let c2 = Complex::new(3.0, 5.0);
   
   // (6+4i)mod(3+5i) = -2+2i
   println!("c1 {:?} c2 {:?} c1 mod c2 {:?}", c1, c2, c1 % c2);
   // Addition Assign
   c1 += c2;
   println!("c1 += c2 {:?}", c1);
   // Subtraction Assign
   c1 -= c2;
   println!("c1 -= c2 {:?}", c1);
   // Multiplication Assign
   c1 *= c2;
   println!("c1 *= c2 {:?}", c1);
   // Division Assign
   c1 /= c2;
   println!("c1 /= c2 {:?}", c1);

   // Norm
   println!("sqr|c1| {:?}", c1.norm());

   // absolute value/modulus/hypotenuse/magnitude
   println!("|c1| {:?}", c1.hypot());

   // Real component
   println!("Real part c1 {:?}", c1.real());

   // Imaginary component
   println!("Imaginery part c1 {:?}", c1.imag());

   // Complex conjugate
   println!("Complex conjugate c1 {:?}", c1.conj());

   // Returns a Complex<T> value from polar coords ( angle in radians )
   println!("Polar to complex number c1 {:?}", Complex::polar(3.0, Complex::degrees_to_radians(45.0)));


   /*The projection of a complex number is a mathematical operation that maps the complex number to the Riemann sphere, often used in complex analysis. Specifically:

   For a complex number z=a+biz=a+bi, the projection is defined as:
   If zz is finite (∣z∣≠∞∣z∣=∞), the projection is zz itself.
   If zz is infinite (∣z∣=∞∣z∣=∞), the projection maps zz to a "point at infinity." */
   println!("Projection c1 {:?} to {:?}", c1, c1.proj());

   // Returns the phase angle (or angular component) of the complex number x, expressed in radians.
   println!("Phase angle c1 {:?} to {:?} radians", c1, c1.arg());
 
   let m: Matrix<i32> = matrix!([
         [1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]
   ]);

   println!("\n\n{:?}", m);

   // Use the matrix! macro to create a Matrix<f64>
   let mut m_complex: Matrix<Complex<f64>> = matrix!([
         [Complex::new(6.0, 4.0), Complex::new(6.0, 4.0)],
         [Complex::new(2.0, 5.0), Complex::new(6.0, 4.0)]
   ]);

   println!("{:?} \nIndex(2,1) {:?}", m_complex, m_complex[(2,1)]);

   m_complex[(2,1)] = Complex::new(7.0, 8.0);

   println!("Index(2,1) {:?}", m_complex[(2,1)]);

   println!("Row 2 {:?}", m_complex.row(2))


}
