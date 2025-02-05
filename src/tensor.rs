use std::ops::{Add, Div, Index, IndexMut, Mul, RangeInclusive, Sub};

pub type TensorCoord = Vec<usize>;

// structure with data stored as row dominant
#[derive(Clone,Debug)]
pub struct Tensor<T> 
{
   dim: Vec<usize>,
   strides: Vec<usize>,
   data: Vec<T>
}

/// Implement indexing of a generic tensor
impl<T> Index<TensorCoord> for Tensor<T>
{
   type Output = T;

   fn index(&self, coord: TensorCoord) -> &T {
      &self.data[coord.iter().zip(self.strides.iter()).fold(0, |acc,(a,b)| acc + a * b )]
   }
}

// /// Implement mutable indexing of a generic matrix; matrix[(i,j,)] = a
// impl<T> IndexMut<(u128,u128)> for Matrix<T>
// {
//    fn index_mut(&mut self, coord:(u128,u128)) -> &mut Self::Output {
//       &mut self.data[((self.cols * (coord.0 - 1)) + coord.1 - 1) as usize]
//    }
// }


/// Implement standard functions for generic matrices
impl<T: Clone + Copy> Tensor<T>
{
   /// Constructor for a new matrix from Vec
   pub fn new(dimensions: TensorCoord, data: &Vec<T> ) -> Self
   where
   T: Clone {
      let mut strides = vec![0 as usize;dimensions.len()];
      for idx in 1..dimensions.len() {
         strides[idx-1] = dimensions.iter().skip(idx).fold(1,|acc,x| acc * *x );
      }
      strides[dimensions.len()-1] = 1;

      let size = dimensions.iter().fold(1,|acc, x| acc * x);
      assert!(
         data.len() == size,
         "The number of elements in the data Vec does not match rows * cols"
      );
      assert!(
         size <= usize::MAX,
         "The number of elements in the data Vec exceeds possible max size usize::MAX"
      );

      Tensor { dim: dimensions,strides: strides, data: data.to_vec() }
   }
}