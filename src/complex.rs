use num_traits::{Float, Num, NumCast, One, Signed, ToPrimitive, Zero};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign,Rem};
use std::fmt::Display;
use std::iter::Sum;
use std::vec::Vec;

// Define a generic Complex structure
#[derive(Debug, Clone, Copy, PartialEq,PartialOrd)]
pub struct Complex<T>
{
   real: T,
   imag: T,
}

/////////////////////////////////////////////////////////////////////////////////////////
/// Implement One,Zero,Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign for Complex\<T\>
///  

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

/// Implement the + trait for Complex\<T\>
///
/// # Arguments
///
/// * `v1` - complex nume.
/// * `v2` - The second vector.
///
/// # Returns
///
/// The dot product of `v1` and `v2`.
///
/// # Example
///
/// ```
/// use crum::complex::Complex;
/// let result = Complex::new(5.8, 3.4) + Complex::new(2.1, 4.0);
/// assert_eq!(result, Complex::new(7.9, 7.4));
/// let result2 = Complex::new(5, 3) + Complex::new(2, 4);
/// assert_eq!(result2, Complex::new(7, 7));
/// ```
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
///
/// # Arguments
///
/// * `v1` - complex nume.
/// * `v2` - The second vector.
///
/// # Returns
///
/// The dot product of `v1` and `v2`.
///
/// # Example
///
/// ```
/// use crum::complex::Complex;
/// let result = Complex::new(5.8, 3.4) * Complex::new(2.1, 4.0);
/// assert_eq!(result, Complex::new(-1.42, 30.34));
/// let result2 = Complex::new(5, 3) * Complex::new(2, 4);
/// assert_eq!(result2, Complex::new(-2, 26));
/// ```
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

/////////////////////////////////////////////////////////////////////////////////////////
/// Standard Functions new,norm, magnitude,real,imag,conj for Complex
/// 
impl<T> Complex<T>
where
   T: Copy
{
   // Constructor for a new complex number
   pub fn new(real: T, imag: T) -> Self 
   {
      Self { real, imag }
   }

   // absolute value/modulus/hypotenuse/magnitude
   pub fn magnitude(&self) -> T
   where 
      T: Float
   {      
      (self.real * self.real + self.imag * self.imag).sqrt()
   }

   #[allow(dead_code)]
   pub fn real(&self) -> T
   {
      self.real
   }

   #[allow(dead_code)]
   pub fn is_real(&self) -> bool
   where 
      T: Zero + PartialEq
   {
      self.imag == T::zero()
   }

   #[allow(dead_code)]
   pub fn imag(&self) -> T
   {
      self.imag
   }

   #[allow(dead_code)]
   pub fn conj(&self) -> Self
   where
   T: Neg<Output = T>
   {
      Self {
         real: self.real,
         imag: -self.imag,
      }
   }

   // Returns a Complex<T> value from polar coords ( angle in radians )
   #[allow(dead_code)]
   pub fn polar(magnitude: T,phase_angle: T ) -> Complex<T>
   where 
      T: Float
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
   #[allow(dead_code)]
   pub fn proj(&self) -> Self
   where 
      T: Float
   {
      let magnitude = self.magnitude();
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

}

/////////////////////////////////////////////////////////////////////////////////////////
/// Implement Float for Complex\<T\>
/// 

impl<T> Float for Complex<T>
where
   T: Float + Num + From<f64>,
   f64: From<T>
{
   fn nan() -> Self {
      Self::new(T::nan(), T::nan())
   }

   fn infinity() -> Self {
      Self::new(T::infinity(), T::zero())
   }

   fn neg_infinity() -> Self {
      Self::new(T::neg_infinity(), T::zero())
   }

   fn neg_zero() -> Self {
      Self::new(T::neg_zero(), T::zero())
   }

   fn min_value() -> Self {
      Self::new(T::min_value(), T::zero())
   }

   fn max_value() -> Self {
      Self::new(T::max_value(), T::zero())
   }

   fn is_nan(self) -> bool {
      self.real.is_nan() || self.imag.is_nan()
   }

   fn is_infinite(self) -> bool {
      self.real.is_infinite() || self.imag.is_infinite()
   }

   fn is_finite(self) -> bool {
      self.real.is_finite() && self.imag.is_finite()
   }

   fn is_normal(self) -> bool {
      self.real.is_normal() && self.imag.is_normal()
   }

   fn classify(self) -> std::num::FpCategory {
      self.real.classify()
   }

   fn floor(self) -> Self {
      Self::new(self.real.floor(), self.imag.floor())
   }

   fn ceil(self) -> Self {
      Self::new(self.real.ceil(), self.imag.ceil())
   }

   fn round(self) -> Self {
      Self::new(self.real.round(), self.imag.round())
   }

   fn trunc(self) -> Self {
      Self::new(self.real.trunc(), self.imag.trunc())
   }

   fn fract(self) -> Self {
      Self::new(self.real.fract(), self.imag.fract())
   }

   fn abs(self) -> Self {
      Self::new(self.magnitude(), T::zero())
   }

   fn signum(self) -> Self {
      let mag = self.magnitude();
      if mag.is_zero() {
         Self::zero()
      } else {
         Self::new(self.real / mag, self.imag / mag)
      }
   }

   fn recip(self) -> Self {
      let mag_sq = self.magnitude();
      Self::new(self.real / mag_sq, -self.imag / mag_sq)
   }

   fn powi(self, n: i32) -> Self {
      let mut result = Self::one();
      let mut base = self;
      let mut exp = n;

      while exp != 0 {
         if exp % 2 != 0 {
               result = result * base;
         }
         base = base * base;
         exp /= 2;
      }
      result
   }

   fn powf(self, _other: Self) -> Self {
      todo!()
   }

   fn sqrt(self) -> Self {
      let mag = self.magnitude().sqrt();
      let angle = self.imag.atan2(self.real) / <T as NumCast>::from(2.0).unwrap();
      Self::new(mag * angle.cos(), mag * angle.sin())
   }

   fn exp(self) -> Self {
      let mag = self.real.exp();
      Self::new(mag * self.imag.cos(), mag * self.imag.sin())
   }

   fn ln(self) -> Self {
      let mag = self.magnitude().ln();
      let angle = self.imag.atan2(self.real);
      Self::new(mag, angle)
   }

   fn sin(self) -> Self {
      Self::new(
         self.real.sin() * self.imag.cosh(),
         self.real.cos() * self.imag.sinh(),
      )
   }

   fn cos(self) -> Self {
      Self::new(
         self.real.cos() * self.imag.cosh(),
         -self.real.sin() * self.imag.sinh(),
      )
   }

   fn tan(self) -> Self {
      //self.sin() / self.cos()
      todo!()
   }
   
   fn min_positive_value() -> Self {
         todo!()
      }

   fn is_sign_positive(self) -> bool {
         todo!()
      }

   fn is_sign_negative(self) -> bool {
         todo!()
      }

   fn mul_add(self, _a: Self, _b: Self) -> Self {
         todo!()
      }

   fn exp2(self) -> Self {
         todo!()
      }

   fn log(self, _base: Self) -> Self {
         todo!()
      }

   fn log2(self) -> Self {
         todo!()
      }

   fn log10(self) -> Self {
         todo!()
      }

   fn max(self, _other: Self) -> Self {
         todo!()
      }

   fn min(self, _other: Self) -> Self {
         todo!()
      }

   fn abs_sub(self, _other: Self) -> Self {
         todo!()
      }

   fn cbrt(self) -> Self {
         todo!()
      }

   fn hypot(self, _other: Self) -> Self {
         todo!()
      }

   fn asin(self) -> Self {
         todo!()
      }

   fn acos(self) -> Self {
         todo!()
      }

   fn atan(self) -> Self {
         todo!()
      }

   fn atan2(self, _other: Self) -> Self {
         todo!()
      }

   fn sin_cos(self) -> (Self, Self) {
         todo!()
      }

   fn exp_m1(self) -> Self {
         todo!()
      }

   fn ln_1p(self) -> Self {
         todo!()
      }

   fn sinh(self) -> Self {
         todo!()
      }

   fn cosh(self) -> Self {
         todo!()
      }

   fn tanh(self) -> Self {
         todo!()
      }

   fn asinh(self) -> Self {
         todo!()
      }

   fn acosh(self) -> Self {
         todo!()
      }

   fn atanh(self) -> Self {
         todo!()
      }

   fn integer_decode(self) -> (u64, i16, i8) {
         todo!()
      }

   fn epsilon() -> Self {
         //Self::from(f32::EPSILON).expect("Unable to cast from f32::EPSILON")
         todo!()
      }

   fn is_subnormal(self) -> bool {
         //self.classify() == std::num::FpCategory::Subnormal
         todo!()
      }

   fn to_degrees(self) -> Self {
      todo!()
      }

   fn to_radians(self) -> Self {
      todo!()
      }

   fn clamp(self, min: Self, max: Self) -> Self {
         num_traits::clamp(self, min, max)
      }

   fn copysign(self, _sign: Self) -> Self {
      todo!()
      }
}

impl<T> Neg for Complex<T>
   where T: Neg<Output = T> + Zero {

      type Output = Self;

   fn neg(self) -> Self {
      Self {
         real: -self.real,
         imag: -self.imag
      }
   }
}

impl<T> NumCast for Complex<T>
where
   T: NumCast + Zero + Float + ToPrimitive,
{
   fn from<U: num_traits::ToPrimitive>(n: U) -> Option<Self> {
      // Cast the input to type T using NumCast
      T::from(n).map(|real| Complex::new(real, T::zero()))
   }
}

impl<T> ToPrimitive for Complex<T>
where
   T: NumCast + Zero + Float + ToPrimitive,
{
   fn to_i64(&self) -> Option<i64> {
      todo!()
   }
   
   fn to_u64(&self) -> Option<u64> {
      todo!()
   }
}

/// Implement the % trait for Complex\<T\>
impl<T> Rem for Complex<T>
where
   T: Clone + Into<f64> + From<f64> + Float + Num
   {
   type Output = Self;
   
   // Division with remainder
   fn rem(self, other: Self) -> Self {
      let norm = other.magnitude();
      let conjugate = other.conj();

      // Compute q = (self * conjugate) / norm
      let real_part = ((self.real.clone() * conjugate.real.clone() - self.imag.clone() * conjugate.imag.clone()) / norm).into();
      let imag_part = ((self.real.clone() * conjugate.imag.clone() + self.imag.clone() * conjugate.real.clone()) / norm).into();

      // Round to the nearest integers
      let rounded_real = real_part.round();
      let rounded_imag = imag_part.round();

      let quotient = Self::new(rounded_real.into(), rounded_imag.into());

      // Compute the remainder: r = self - (quotient * other)
      let product = Self::new(
         quotient.real.clone() * other.real.clone() - quotient.imag.clone() * other.imag.clone(),
         quotient.real.clone() * other.imag.clone() + quotient.imag.clone() * other.real.clone(),
      );

      let remainder = Self::new(self.real.clone() - product.real.clone(), self.imag.clone() - product.imag.clone());

      remainder
   }
   
}


// Implement `Num` for `Complex\<T\>`
impl<T> Num for Complex<T>
where
   T: Float + Num + Copy + std::convert::From<f64>,
   f64: From<T>,
{
   type FromStrRadixErr = T::FromStrRadixErr;

   fn from_str_radix(s: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
      // Parse a string in the form "a+bj" or "a-bj"
      let parts: Vec<&str> = s.split(|c| c == '+' || c == '-').collect();
      let real = T::from_str_radix(parts[0], radix)?;
      let imag = if s.contains('-') {
         -T::from_str_radix(parts[1], radix)?
      } else {
         T::from_str_radix(parts[1], radix)?
      };
      Ok(Self::new(real, imag))
   }
}

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


impl<T: Display + Clone + Signed + PartialOrd> Display for Complex<T> {
   fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
      if self.imag < T::zero() {
         if self.real < T::zero() {
            write!(f, "{} - i{}", format!("{:5.4}",self.real.clone()), format!("{:4.4}",num_traits::abs(self.imag.clone()))).expect("Not Written"); 
         } else {
            write!(f, " {} - i{}", format!("{:4.4}",self.real.clone()), format!("{:4.4}",num_traits::abs(self.imag.clone()))).expect("Not Written");
         }
      } else {
         if self.real < T::zero() {
            write!(f, "{} + i{}", format!("{:5.4}",self.real.clone()), format!("{:4.4}",num_traits::abs(self.imag.clone()))).expect("Not Written"); 
         } else {
            write!(f, " {} + i{}", format!("{:4.4}",self.real.clone()), format!("{:4.4}",num_traits::abs(self.imag.clone()))).expect("Not Written");
         }
      }
      Ok(())
   }
}
