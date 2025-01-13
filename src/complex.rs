use num_traits::{Float, PrimInt,Zero,One,Num};
use std::ops::{Add, AddAssign, Sub,SubAssign, Mul, MulAssign,Div,DivAssign,Neg,Rem,RemAssign};

// Define a generic Complex structure
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex<T> {
   real: T,
   imag: T,
}

// Implement the Zero trait for Complex<T>
impl<T> Zero for Complex<T>
where
   T: PrimInt + Float,
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
   T: PrimInt + Float,
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

// Implement the % trait for Complex<T> T as Float
impl<T> Rem for Complex<T>
where
   T: Float
{
   type Output = Self;

   fn rem(self, other: Self) -> Self {
      let norm = other.norm();
      let conjugate = other.conj();

      // Compute q = (self * conjugate) / norm
      let real_part = (self.real * conjugate.real - self.imag * conjugate.imag) / norm;
      let imag_part = (self.real * conjugate.imag + self.imag * conjugate.real) / norm;

      // Round to nearest integers
      let rounded_real = real_part.round();
      let rounded_imag = imag_part.round();

      let quotient = Complex::new(rounded_real, rounded_imag);

      let product = Complex::new(
         quotient.real * other.real - quotient.imag * other.imag,
         quotient.real * other.imag + quotient.imag * other.real,
      );

      Self{
         real: self.real - product.real,
         imag: self.imag - product.imag,
      }
   }
}


// Implement the += trait for Complex<T>
impl<T> AddAssign for Complex<T>
where
   T: AddAssign,
{
   fn add_assign(&mut self, other: Self) {
      self.real += other.real;
      self.imag += other.imag;
   }
}
// Implement the - trait for Complex<T>
impl<T> Sub for Complex<T>
where
   T: Sub<Output = T>,
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
   T: SubAssign,
{
   fn sub_assign(&mut self, other: Self) {
      self.real -= other.real;
      self.imag -= other.imag;
   }
}

// Implement * (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
impl<T> Mul for Complex<T>
where
   T: Mul<Output = T> + Sub<Output = T> + Add<Output = T> + Clone
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
   T: Mul<Output = T> + Sub<Output = T> + Add<Output = T> + Clone,
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
   T: Div<Output= T> + Mul<Output = T> + Sub<Output = T> + Add<Output = T> + Clone,
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
   T: Mul<Output = T> + Sub<Output = T> + Add<Output = T> + Div<Output = T> + Clone
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
impl<T> Complex<T>
{
   // Constructor for a new complex number
   pub fn new(real: T, imag: T) -> Self {
      Self { real, imag }
   }

   // absolute value/modulus/hypotenuse/magnitude
   pub fn norm(&self) -> T 
   where
      T: Clone + Add<Output = T> + Mul<Output = T>
   {      
      self.real.clone() * self.real.clone() + self.imag.clone() * self.imag.clone()
   }

   // absolute value/modulus/hypotenuse/magnitude
   pub fn hypot(&self) -> T 
      where
         T: Float + Clone
   {      
      self.real.clone().hypot(self.imag.clone())
   }

   // 
   pub fn real(&self) -> T
      where
         T: Clone    
   {
      self.real.clone()
   }

   pub fn imag(&self) -> T
      where
         T: Clone        
   {
      self.imag.clone()
   }

   pub fn conj(&self) -> Self
      where
         T: Neg<Output = T> + Clone,
   {
      Self {
         real: self.real.clone(),
         imag: -self.imag.clone(),
      }
   }

   pub fn degrees_to_radians(degrees: T) -> T
      where
         T: Float,
   {
      let pi = T::from(std::f64::consts::PI).unwrap(); // Convert PI to type T
      degrees * pi / T::from(180.0).unwrap()
   }

   // Returns a Complex<T> value from polar coords ( angle in radians )
   pub fn polar(magnitude: T,phase_angle: T ) -> Complex<T>
   where
      T: Num + Float + Clone,
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
      where
         T: Float + Num + Clone,
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

   // Returns the phase angle (or angular component) of the complex number x, expressed in radians.  
   pub fn arg(&self) -> T
      where
         T: Num + Float + Clone,
   {
      self.imag.clone().atan2(self.real.clone())
   }
}

