use num_traits::{Float,Num,Zero,One};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::fmt::Display;

// Define a generic Complex structure
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex<T>
{
   real: T,
   imag: T,
}

// Implement the Zero trait for Complex<T>
impl<T> Zero for Complex<T>
where
   T: Zero + PartialEq,
{
   fn zero() -> Self {
      Self {
         real: T::zero(),
         imag: T::zero(),
      }
   }
   fn is_zero(&self) -> bool {
      if self.real == T::zero() && self.imag == T::zero() {
         true
      } else {
         false
      }
   }
}

// Implement the One trait for Complex<T>
impl<T> One for Complex<T>
where
   T: Clone + Sub<Output = T> + One + Zero + PartialEq,
{
   fn one() -> Self {
      Self {
         real: T::one(),
         imag: T::zero(),
      }
   }
   fn is_one(&self) -> bool {
      if self.real == T::one() && self.imag == T::zero() {
         true
      } else {
         false
      }
   }
}

// Implement the + trait for Complex<T>
impl<T> Add for Complex<T>
where
   T: Add<Output = T>
{
   type Output = Self;

   fn add(self, other: Self) -> Self {
      Self {
         real: self.real + other.real,
         imag: self.imag + other.imag,
      }
   }
}

// Implement the += trait for Complex<T>
impl<T> AddAssign for Complex<T>
where
   T: AddAssign
{
   fn add_assign(&mut self, other: Self) {
      self.real += other.real;
      self.imag += other.imag;
   }
}

// Implement the - trait for Complex<T>
impl<T> Sub for Complex<T>
where
   T: Sub<Output = T>
{
   type Output = Self;

   fn sub(self, other: Self) -> Self {
      Self {
         real: self.real - other.real,
         imag: self.imag - other.imag,
      }
   }
}

// Implement the -= trait for Complex<T>
impl<T> SubAssign for Complex<T>
where
   T: SubAssign
{
   fn sub_assign(&mut self, other: Self) {
      self.real -= other.real;
      self.imag -= other.imag;
   }
}

// Implement * (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
impl<T> Mul for Complex<T>
where
   T: Clone + Mul<Output = T> + Add<Output = T> + Sub<Output = T>
{
   type Output = Self;

   fn mul(self, other: Self) -> Self {
      Self {
         real: self.real.clone() * other.real.clone() - self.imag.clone() * other.imag.clone(),
         imag: self.real * other.imag + self.imag * other.real,
      }
   }
}

// Implement *= trait for Complex<T>
impl<T> MulAssign for Complex<T>
where
   T: Clone + Mul<Output = T> + Add<Output = T> + Sub<Output = T>
{
   fn mul_assign(&mut self, other: Self) {
      let real = self.real.clone() * other.real.clone() - self.imag.clone() * other.imag.clone();
      let imag = self.real.clone() * other.imag + self.imag.clone() * other.real;
      self.real = real;
      self.imag = imag;      
   }
}

// Implement / z1 = a + bi,z2 = c + di then z1 / z2 = [(a * c + b * d) + (b * c - a * d)i] / (c² + d²)
impl<T> Div for Complex<T>
where
   T: Clone + Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Div<Output = T> 
{
   type Output = Self;

   fn div(self, other: Self) -> Self {
      let denom = other.real.clone() * other.real.clone() + other.imag.clone() * other.imag.clone();
      Self {         
         real: (self.real.clone() * other.real.clone() + self.imag.clone() * other.imag.clone())/denom.clone(),
         imag: (self.imag * other.real - self.real * other.imag)/denom,
      }
   }
}

// Implement *= trait for Complex<T>
impl<T> DivAssign for Complex<T>
where
   T: Clone + Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Div<Output = T> 
{
   fn div_assign(&mut self, other: Self) {
      let denom = other.real.clone() * other.real.clone() + other.imag.clone() * other.imag.clone();
      let real = (self.real.clone() * other.real.clone() + self.imag.clone() * other.imag.clone())/denom.clone();
      let imag = (self.imag.clone() * other.real - self.real.clone() * other.imag)/denom;
      self.real = real;
      self.imag = imag;      
   }
}

// Implement standard functions for complex numbers
impl<T: Clone> Complex<T>
{
   // Constructor for a new complex number
   pub fn new(real: T, imag: T) -> Self 
   {
      Self { real, imag }
   }

   // absolute value/modulus/hypotenuse/magnitude
   pub fn norm(&self) -> T
   where
      T: Mul<Output = T> + Add<Output = T> 
   {      
      self.real.clone() * self.real.clone() + self.imag.clone() * self.imag.clone()
   }

   // 
   pub fn real(&self) -> T
   {
      self.real.clone()
   }

   pub fn imag(&self) -> T
   {
      self.imag.clone()
   }

   pub fn conj(&self) -> Self
   where
   T: Neg<Output = T>
   {
      Self {
         real: self.real.clone(),
         imag: -self.imag.clone(),
      }
   }


}

// Implement standard functions for complex numbers
impl<T: Clone + Float> Complex<T>
{
   // absolute value/modulus/hypotenuse/magnitude
   pub fn hypot(&self) -> T 
   {      
      self.real.clone().hypot(self.imag.clone())
   }

   pub fn degrees_to_radians(degrees: T) -> T
   {
      let pi = T::from(std::f64::consts::PI).unwrap(); // Convert PI to type T
      degrees * pi / T::from(180.0).unwrap()
   }

   // Returns a Complex<T> value from polar coords ( angle in radians )
   pub fn polar(magnitude: T,phase_angle: T ) -> Complex<T>
   {
      Complex { real: magnitude * phase_angle.cos(), imag: magnitude * phase_angle.sin()}
   }

   /*The projection of a complex number is a mathematical operation that maps the complex number to the Riemann sphere, often used in complex analysis. Specifically:

   For a complex number z=a+biz=a+bi, the projection is defined as:
   If zz is finite (∣z∣≠∞∣z∣=∞), the projection is zz itself.
   If zz is infinite (∣z∣=∞∣z∣=∞), the projection maps zz to a "point at infinity."

   In computational terms, the projection can be approximated by:

   Returning zz as is if it is finite.
   Returning "infinity" when the magnitude exceeds a certain threshold. */
   pub fn proj(&self) -> Self
   {
      let magnitude = self.hypot();
      let infinity = T::infinity();

      if magnitude.is_infinite() {
         // If the magnitude is infinite, map to a "point at infinity"
         Self {
            real: infinity,
            imag: T::zero(), // Imaginary part is zero at infinity
         }
      } else {
         // Otherwise, return the number as is
         self.clone()
      }
   }

   pub fn sqrt(&self) -> Self
   {
      let magnitude = self.hypot();
      let two = T::from(2.0);

      let real_part = ((magnitude + self.real) / two.unwrap()).sqrt();
      let imag_sign = if self.imag < T::from(0.0).unwrap() { -T::from(1.0).unwrap() } else { T::from(1.0).unwrap() };
      let imag_part = imag_sign * ((magnitude - self.real) / two.unwrap()).sqrt();

      Self::new(real_part, imag_part)      
   }   

   // Returns the phase angle (or angular component) of the complex number x, expressed in radians.  
   pub fn arg(&self) -> T
   {
      self.imag.clone().atan2(self.real.clone())
   }

}

use std::iter::Sum;

impl<T> Sum for Complex<T>
   where T: Add<Output = T> + Zero {
   fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
      iter.fold(
         Complex { real: T::zero(), imag: T::zero() },
         |acc, x| Complex {
               real: acc.real + x.real,
               imag: acc.imag + x.imag,
         },
      )
   }
}


impl<T: Display + Clone + num_traits::Signed + std::cmp::PartialOrd> Display for Complex<T> {
   fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
      if self.imag < T::zero() {
         write!(f, "{}-i{}", self.real.clone(), num_traits::abs(self.imag.clone())).expect("Not Written"); 
      } else {
         write!(f, "{}+i{}", self.real.clone(),self.imag.clone()).expect("Not Written");
      }
      Ok(())
   }
}


