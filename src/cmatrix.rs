use crate::complex::Complex;
use std::ops::{Add,Index,IndexMut,Mul,Sub, Div};
use std::vec::Vec;
use std::fmt::Display;
use num_traits::{Zero,Float};

//
// Some linear alg operations on a vector required
//
/// Compute the dot product of two vectors
fn dot_product<T>(v1: &Vec<T>, v2: &Vec<T>) -> T
where
   T: Copy + Zero + Mul<Output = T>
{
   v1.iter().zip(v2).fold(T::zero(), |acc, (&x, &y)| acc + x * y)
}

/// Compute the 2-norm of a vector
fn col_norm<T>(v: &Vec<T>) -> T
where
   T: Copy + Float
{
   let sum = v.iter().fold(T::zero(), |acc, &x| acc + x * x);
   sum.sqrt()
}

/// Compute the scaler * vector
fn scalar_mul<T>(v: &Vec<T>, a:T) -> Vec<T>
where
   T: Clone + Mul<Output = T>
{
   v.iter().map(|x| a.clone() * x.clone()).collect()
}

/// Compute the vector / scalar
fn scalar_div<T>(v: &Vec<T>, a:T) -> Vec<T>
where
   T: Clone + Div<Output = T>
{
   v.iter().map(|x| x.clone()/a.clone()).collect()
}

/// Compute vector - vector
fn vector_sub<T>(v1: &Vec<T>, v2: &Vec<T>) -> Vec<T>
where
   T: Copy + Sub<Output = T>
{
   v1.iter().zip(v2).map(|(&x,&y)| x - y).collect()
}

// Define a generic matrix structure
// with data stored as row dominant
#[derive(Debug, Clone)]
pub struct Matrix<T> 
{
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

impl<T> Add<T> for Matrix<T>
   where
      T: Copy + Add<Output = T>,
{
   type Output = Self;

   fn add(self, other: T) -> Self {
      Self {
         rows: self.rows,
         cols: self.cols,
         data: self.data.iter().map(|&x| x+other).collect()
      }
   }
}

//Implement a = matrix[(i,j)] index and matrix[(i,j,)] = a
impl<T> Index<(u128,u128)> for Matrix<T>
   {
      type Output = T;

      fn index(&self, coord:(u128,u128)) -> &T {
         &self.data[((self.cols * (coord.0 - 1)) + coord.1 - 1) as usize]
      }
   }

impl<T> IndexMut<(u128,u128)> for Matrix<T>
   {
      fn index_mut(&mut self, coord:(u128,u128)) -> &mut Self::Output {
         &mut self.data[((self.cols * (coord.0 - 1)) + coord.1 - 1) as usize]
      }
   }

// Implement == comparison for matrices
impl<T> PartialEq for Matrix<T>
   where
   T: PartialEq
{
   fn eq(&self, other: &Self) -> bool {
      assert_eq!((self.rows,self.cols),(other.rows,other.cols),"Comparison of matrices requires the same dimensions.");
      
      let data:Vec<bool> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a == b)
            .collect();
      
      !data.iter().any(|&x| !x)
         
   }
}



// Implement standard functions for complex numbers
impl<T: Clone> Matrix<T>
{
   // Constructor for a new matrix from Vec
   pub fn new(rows: u128, cols: u128, data: Vec<T> ) -> Self
   where
   T: Clone {
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

   // Get nth row as Vec<T>
   pub fn row(&self, idx: u128) -> &[T] {
      &self.data[(self.cols*(idx-1)) as usize..((self.cols*(idx-1)) + self.cols) as usize]
   }

   // Get nth col as Vec<T>
   pub fn col(&self, idx: u128) -> Vec<T> {
                        self
                        .data
                        .iter()                  // Create an iterator over the vector
                        .skip((idx - 1) as usize)                 // Skip the first elements
                        .step_by(self.cols as usize)              // Take every nth element
                        .cloned()
                        .collect()
                     
   }

   // Get nth col as Vec<T>
   pub fn col_set(self, idx: u128, col: Vec<T>) -> Self
   where 
      {
      let mut data= self.data;      
      let vec_iter = data
      .iter_mut()  // mutable iterator over the vector
      .skip((idx - 1) as usize)  // Skip the first elements
      .step_by(self.cols as usize);
      
      let mut index:usize = 0;
      for x in vec_iter{
         let temp = &col[index];
         *x = col[index].clone();
         index += 1;
      }

      Self {
         rows: self.rows,
         cols: self.cols,
         data: data.to_vec()
      }
   
}

   // Get data Vec 
   pub fn data(&self) -> Vec<T> {
      self.data.clone()
   }

   // Transpose
   pub fn trans(self) -> Self {   

      let mut result_vec: Vec<T> = Vec::with_capacity((self.rows*self.cols) as usize);
      for row_idx in 1..=self.cols{
         let iter = SkipIter::new(&self.data, (row_idx - 1) as usize, self.cols as usize);
         let current_row:Vec<T> = iter.map(|x| x.clone()).collect();
         result_vec.extend(current_row);
      }

      Self {
         rows: self.cols,
         cols: self.rows,
         data: result_vec
      }
   }
}



impl<T> Matrix<T> {
   // QR Decomposition - Gram-Schmidt
   pub fn qr_decomp_gs(&self) ->(Self,Self)
      where
         T:Copy + Zero + Float
         {
            let mut q = Matrix::new(self.rows, self.cols, vec![T::zero(); (self.rows*self.cols) as usize ]);
            let mut r = Matrix::new(self.rows, self.cols, vec![T::zero(); (self.rows*self.cols) as usize ]);
            // For each column in self
            // Define vector as a n x 1 matrix
            for i in 1..=self.cols {
               let mut col_i = self.col(i);

               // Orthogonalize the current column against all preceding columns
               for j in 1..=i {
                  let col_j = self.col(j);
                  let r_ji = dot_product(&col_j, &col_i);

                  r[(j,i)] = r_ji; 
                  
                  col_i = vector_sub(&col_j,&scalar_mul(&col_j,r_ji));

               }

               // Normalize column
               let norm = col_norm(&col_i);
               r[(i,i)] = norm; 
               col_i = scalar_div(&col_i,norm);
            
               // set as ith column of Q matrix
               q = q.clone().col_set(i, col_i);
            }
            (q,r)
         }



}


// Complex Matrix Specializations
impl<T> Matrix<Complex<T>>
   where
      Matrix<Complex<T>>: PartialEq,
      T: Clone + Float + std::ops::Neg<Output = T>
{
   
   // Complex Conjugate
   pub fn conj(self) -> Self {
      Self {
         rows: self.cols,
         cols: self.rows,
         data: self.data.iter().map(|x| Complex::new(x.real(),-x.imag())).collect()
      }
   }

   // Hermitian/Self-adjoint - Charles Hermite 1855
   pub fn is_hermitian(self) -> bool {
      if self.clone().trans().conj() == self {
         true
      } else {
         false
      }
   }

}

#[macro_export]
#[allow(unused_assignments)]
macro_rules! matrix {
   // Match rows and columns   
   ( $(  [$($x:expr),* ]   ),*) => {
      {
         let mut data = Vec::<_>::new();
         let mut rows : u128 = 0;
         let mut first_row_cols : usize = 0;
         let mut row_cols: usize = 0;
         let mut is_first_row: bool = true;
         $(
            rows += 1;            
            $(
               if is_first_row {
                  first_row_cols += 1;
               } else {
                  row_cols += 1;
               }
               data.push($x);               
               //println!("{:?}",$x);
            )*
            if is_first_row {
               is_first_row = false;
            } else{
               assert_eq!(first_row_cols as usize, row_cols, "All rows must have the same number of columns");
            }       
         )*
         Matrix::new(rows, first_row_cols as u128, data)  
       }
   };
   
   // ( $rows:expr; $cols:expr; $val:expr; $t:ty ) => {{
   //          let data = Vec<$t>::from_element($val,$rows * $cols);
   //          //println!("Rows {} Cols {}", rows, first_row_cols);
   //          Matrix::new($rows, $cols, data)
   // }};
}

//
// Matrix Multiplication : matrix * matrix, matrix * scalar
//

// Define a custom skip iterator
pub struct SkipIter<'a, T> {
   vec: &'a [T],      // Reference to the slice (borrowed from the Vec)
   index: usize,      // Current index
   step: usize,       // Step/offset to skip
}

impl<'a, T> SkipIter<'a, T> {
   // Create a new SkipIter
   pub fn new(vec: &'a Vec<T>, start: usize, step: usize) -> Self {
      Self {
           vec: &vec[start..], // Slice starting at `start` index
           index: 0,           // Initialize current index
           step,               // Store the step size
      }
   }
}

// Implement the Iterator trait for SkipIter
impl<'a, T> Iterator for SkipIter<'a, T> {
   type Item = &'a T;

   fn next(&mut self) -> Option<Self::Item> {
      if self.index >= self.vec.len() {
           None // Stop iteration if we're out of bounds
      } else {
           let item = &self.vec[self.index]; // Get the current item
           self.index += self.step;          // Move to the next step
         Some(item)
      }
   }
}

impl<T> Mul for Matrix<T>
   where
      T: Copy + std::iter::Sum + Mul<Output = T>
{
   type Output = Self;

   fn mul(self, other: Self) -> Self {
      assert_eq!(self.cols, other.rows, "Multiplication requires LHS col dimension equal RHS row dimension");

      let mut result_vec: Vec<T> = Vec::new();
      for row_idx in 1..=self.rows{
         for col_idx in 1..=other.cols {
            let current_row = (&self.data[(self.cols*(row_idx-1)) as usize..((self.cols*(row_idx-1)) + self.cols) as usize]).to_vec();
            let iter = SkipIter::new(&other.data, (col_idx - 1) as usize, other.cols as usize);
            let current_col = iter.map(|&x| x).collect::<Vec<T>>();
            result_vec.push(current_row.into_iter().zip(current_col.into_iter()).map(|(x, y)| x * y).sum());
         }
      }

      Self {
         rows: self.rows,
         cols: other.cols,
         data: result_vec
      }
   }
}

impl<T> Mul<T> for Matrix<T>
where 
   T: Copy + Mul<Output = T>
{
   type Output = Self;

   fn mul(self, other: T) -> Self {
      Self {
         rows: self.rows,
         cols: self.cols,
         data: self.data.iter().map(|&x| x*other).collect()
      }
   }
}

// Display a matrix sensibly
impl<T: Display +Clone> Display for Matrix<T> {
   fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
      write!(f, "\n[").expect("Not Written");
      for row_idx in 1..=self.rows{         
         write!(f, "[").expect("Not Written");
         for col_idx in 1..=self.cols {
            write!(f, "\t{}", self[(row_idx,col_idx)]).expect("Not Written");
         }
         if row_idx == self.rows {
            write!(f, "\t]").expect("Not Written");
         } else {
            write!(f, "\t]\n").expect("Not Written");
         }
         
      }
      write!(f, "]").expect("Not Written");
      Ok(())
   }
}
