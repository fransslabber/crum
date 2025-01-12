use num_traits::{Float, Num};
use std::ops::{Add, AddAssign, Sub,SubAssign, Mul, MulAssign,Div,DivAssign,Neg,Index,IndexMut};
use std::vec::Vec;

// Define a generic matrix structure
// with data stored as row dominant
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T> {
   rows: u128,
   cols: u128,
   data: Vec<T>
}

// Implement the + trait for Matrix<T>
impl<T> Add for Matrix<T>
   where
      T: Clone + Add<Output = T>,
   {
      type Output = Result<Self,&'static str>;

      fn add(self, other: Self) -> Result<Self, &'static str> {
      if self.rows == other.rows && self.cols == other.cols {
            let data = self
               .data
               .iter()
               .zip(other.data.iter())
               .map(|(a, b)| a.clone() + b.clone())
               .collect();
            Ok(Self {
               rows: self.rows,
               cols: self.cols,
               data,
            })
      } else {
            Err("Addition requires matrices of the same dimensions")
      }
   }
}
// // Implement the + trait for Matrix<T>
// impl<T> Index<(u128,u128)> for Matrix<T>
//    where
//       T: Clone,
//    {
//       type Output = T;

//       fn index(self, row, col) -> T {

//       }
//    }

// // Implement the + trait for Matrix<T>
// impl<T,Idx> IndexMut<Idx> for Matrix<T>
//    where
//       T: Clone + Add<Output = T>,
//    {
//       type Output = Result<Self,&'static str>;

//       fn index_mut(self, other: Self) -> Result<Self, &'static str> {
//       if self.cols == other.rows {
//             let data = self
//                .data
//                .iter()
//                .zip(other.data.iter())
//                .map(|(a, b)| a.clone() + b.clone())
//                .collect();
//             Ok(Self {
//                rows: self.rows,
//                cols: self.cols,
//                data,
//             })
//       } else {
//             Err("Multiplicaton of matrices A*B requires A column dimension = B row dimension")
//       }
//    }
// }

// // Implement the += trait for Complex<T>
// impl<T> AddAssign for Complex<T>
// where
//    T: AddAssign,
// {
//    fn add_assign(&mut self, other: Self) {
//       self.real += other.real;
//       self.imag += other.imag;
//    }
// }
// // Implement the - trait for Complex<T>
// impl<T> Sub for Complex<T>
// where
//    T: Sub<Output = T>,
// {
//    type Output = Self;

//    fn sub(self, other: Self) -> Self {
//       Self {
//          real: self.real - other.real,
//          imag: self.imag - other.imag,
//       }
//    }
// }

// // Implement the -= trait for Complex<T>
// impl<T> SubAssign for Complex<T>
// where
//    T: SubAssign,
// {
//    fn sub_assign(&mut self, other: Self) {
//       self.real -= other.real;
//       self.imag -= other.imag;
//    }
// }

// Implement * (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
// impl<T> Mul for Matrix<T>
// where
//    T: Mul<Output = T> + Sub<Output = T> + Add<Output = T> + Clone
// {
//    type Output = Self;

//    fn mul(self, other: Self) -> Self {
//       Self {
//          real: self.real.clone() * other.real.clone() - self.imag.clone() * other.imag.clone(),
//          imag: self.real * other.imag + self.imag * other.real,
//       }
//    }
// }

// // Implement *= trait for Complex<T>
// impl<T> MulAssign for Complex<T>
// where
//    T: Mul<Output = T> + Sub<Output = T> + Add<Output = T> + Clone,
// {
//    fn mul_assign(&mut self, other: Self) {
//       let real = self.real.clone() * other.real.clone() - self.imag.clone() * other.imag.clone();
//       let imag = self.real.clone() * other.imag + self.imag.clone() * other.real;
//       self.real = real;
//       self.imag = imag;      
//    }
// }

// // Implement / z1 = a + bi,z2 = c + di then z1 / z2 = [(a * c + b * d) + (b * c - a * d)i] / (c² + d²)
// impl<T> Div for Complex<T>
// where
//    T: Mul<Output = T> + Sub<Output = T> + Add<Output = T> + Div<Output = T>+ Clone
// {
//    type Output = Self;

//    fn div(self, other: Self) -> Self {
//       let denom = other.real.clone() * other.real.clone() + other.imag.clone() * other.imag.clone();
//       Self {         
//          real: (self.real.clone() * other.real.clone() + self.imag.clone() * other.imag.clone())/denom.clone(),
//          imag: (self.imag * other.real - self.real * other.imag)/denom,
//       }
//    }
// }

// // Implement *= trait for Complex<T>
// impl<T> DivAssign for Complex<T>
// where
//    T: Mul<Output = T> + Sub<Output = T> + Add<Output = T> + Div<Output = T>+ Clone
// {
//    fn div_assign(&mut self, other: Self) {
//       let denom = other.real.clone() * other.real.clone() + other.imag.clone() * other.imag.clone();
//       let real = (self.real.clone() * other.real.clone() + self.imag.clone() * other.imag.clone())/denom.clone();
//       let imag = (self.imag.clone() * other.real - self.real.clone() * other.imag)/denom;
//       self.real = real;
//       self.imag = imag;      
//    }
// }

// Implement standard functions for complex numbers
impl<T: Clone> Matrix<T>
{
   // Constructor for a new matrix from Vec
   pub fn new(rows: u128, cols: u128, data: Vec<T> ) -> Self {
      let size = (rows * cols) as usize;
      assert!(
         data.len() == size,
         "The number of elements in the data Vec does not match rows * cols"
      );
      assert!(
         size <= usize::MAX,
         "The number of elements in the data Vec exceeds possible max size usize::MAX"
      );

      Matrix { rows, cols, data }
   }

}

#[macro_export]
macro_rules! matrix {
   // Match rows and columns
   
   ( $(  [$($x:expr),* ]   ),*) => {
      {
         // Collect all elements into a flat vector
         let mut data = Vec::new();
         let mut rows : u128 = 0;
         let mut first_row_cols : usize = 0;
         let mut is_first_row: bool = true;
         $(
            $(
               rows += 1;
               if is_first_row {
                  first_row_cols = $x.len();
                  is_first_row = false;
               } else{
                  assert_eq!(first_row_cols as usize, $x.len(), "All rows must have the same number of columns");
               }           
               data.extend($x);
            )*
         )*
         Matrix::new(rows, first_row_cols as u128, data)
      }
   };
}

//    // absolute value/modulus/hypotenuse
//    pub fn hypot(&self) -> T 
//       where
//          T: Float + Clone
//    {      
//       self.real.clone().hypot(self.imag.clone())
//    }

//    // 
//    pub fn real(&self) -> T
//       where
//          T: Clone    
//    {
//       self.real.clone()
//    }

//    pub fn imag(&self) -> T
//       where
//          T: Clone        
//    {
//       self.imag.clone()
//    }

//    pub fn conj(&self) -> Self
//       where
//          T: Neg<Output = T> + Clone,
//    {
//       Self {
//          real: self.real.clone(),
//          imag: -self.imag.clone(),
//       }
//    }

//    pub fn degrees_to_radians(degrees: T) -> T
//       where
//          T: Float,
//    {
//       let pi = T::from(std::f64::consts::PI).unwrap(); // Convert PI to type T
//       degrees * pi / T::from(180.0).unwrap()
//    }

//    // Returns a Complex<T> value from polar coords ( angle in radians )
//    pub fn polar(magnitude: T,phase_angle: T ) -> Complex<T>
//    where
//       T: Float + Num + Clone,
//    {
//       Complex { real: magnitude * phase_angle.cos(), imag: magnitude * phase_angle.sin()}
//    }

//    /*The projection of a complex number is a mathematical operation that maps the complex number to the Riemann sphere, often used in complex analysis. Specifically:

//    For a complex number z=a+biz=a+bi, the projection is defined as:
//    If zz is finite (∣z∣≠∞∣z∣=∞), the projection is zz itself.
//    If zz is infinite (∣z∣=∞∣z∣=∞), the projection maps zz to a "point at infinity."

//    In computational terms, the projection can be approximated by:

//    Returning zz as is if it is finite.
//    Returning "infinity" when the magnitude exceeds a certain threshold. */
//    pub fn proj(&self) -> Self
//       where
//          T: Float
//    {
//       let magnitude = self.hypot();
//       let infinity = T::infinity();

//       if magnitude.is_infinite() {
//          // If the magnitude is infinite, map to a "point at infinity"
//          Self {
//             real: infinity,
//             imag: T::zero(), // Imaginary part is zero at infinity
//          }
//       } else {
//          // Otherwise, return the number as is
//          self.clone()
//       }
//    }

//    // Returns the phase angle (or angular component) of the complex number x, expressed in radians.  
//    pub fn arg(&self) -> T
//       where
//          T: Float
//    {
//       self.imag.clone().atan2(self.real.clone())
//    }
// }

