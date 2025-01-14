use std::ops::{Add,Index,IndexMut,Mul};
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

   // Get data Vec 
   pub fn data(&self) -> Vec<T> {
      self.data.clone()
   }

   // Get scalar multiplication
   pub fn mul_scalar(&self, num: &T) -> Self 
      where T: Mul<Output = T> + Clone 
      {
      let data: Vec<T> = self
      .data
      .iter()                  // Create an iterator over the vector
      .map(|x| x.clone()*num.clone() )
      .collect();

      Self {
         rows:self.rows,
         cols: self.cols,
         data: data
      }
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

//
// Matrix Multiplication
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
      T: Copy + Mul<Output = T>+ Add<Output = T> + std::iter::Sum
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