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
   pub fn col(&self, idx: u128) -> &[T] {
                        let d =&self
                        .data
                        .iter()                  // Create an iterator over the vector
                        .skip(idx as usize)                 // Skip the first 3 elements
                        .step_by(self.cols as usize)              // Take every 5th element
                        .collect();
                     d
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

// impl<T> Mul for Matrix<T>
//    where
//       T: Clone + Mul<Output = T>,
// {
//    type Output = Result<Self,&'static str>;

//    fn mul(self, other: Self) -> Result<Self, &'static str> {
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
//             Err("Multiplication requires LHS col dimension equal RHS row dimension")
//       }
//    }
// }