use std::fmt::Display;
use std::ops::{Add, Div, Index, IndexMut, Mul, Range, RangeInclusive, Sub};
use num_traits::{Float, NumCast, One, Signed, ToPrimitive, Zero};
use rand::distributions::uniform::SampleUniform;
use rand::Rng;

// structure with data stored as row dominant
#[derive(Clone,Debug)]
pub struct Tensor<T> 
{
   shape: Vec<usize>,
   strides: Vec<usize>,
   data: Vec<T>
}

/// Implement indexing of a generic tensor
impl<T> Index<&Vec<usize>> for Tensor<T>
{
   type Output = T;

   fn index(&self, coord: &Vec<usize>) -> &T {
      &self.data[coord.iter().zip(self.strides.iter()).fold(0, |acc,(a,b)| acc + a * b )]
   }
}

/// Implement mutable indexing of a generic matrix; matrix[(i,j,)] = a
impl<T> IndexMut<&Vec<usize>> for Tensor<T>
{
   fn index_mut(&mut self, coord: &Vec<usize>) -> &mut Self::Output {
      &mut self.data[coord.iter().zip(self.strides.iter()).fold(0, |acc,(a,b)| acc + a * b )]
   }
}


/// Implement standard functions for generic tensors
impl<T: Clone + Copy> Tensor<T>
{
   /// Constructor for a new matrix from Vec
   pub fn new(shape: Vec<usize>, data: &Vec<T> ) -> Self
   {
      let mut strides = vec![0 as usize;shape.len()];
      for idx in 1..shape.len() {
         strides[idx-1] = shape.iter().skip(idx).fold(1,|acc,x| acc * *x );
      }
      strides[shape.len()-1] = 1;

      let size = shape.iter().fold(1,|acc, x| acc * x);
      assert!(
         data.len() == size,
         "The number of elements in the data Vec does not match dimension element count"
      );
      assert!(
         size <= usize::MAX,
         "The number of elements in the data Vec exceeds possible max size usize::MAX"
      );

      Tensor { shape, strides, data: data.to_vec() }
   }
   
   /// Extract a sub tensor from a tensor using inclusive ranges.
   /// ```
   /// use crum::tensor;
   /// let t3 = tensor![[
   ///   [
   ///         [1.0, 2.0, 3.0],
   ///         [4.0, 5.0, 6.0]
   ///   ],
   ///   [
   ///         [7.0, 8.0, 9.0],
   ///         [10.0, 11.0, 12.0]
   ///   ]
   /// ]];
   /// let t2 = t3.subtensor(&vec![1..=1,0..=1,1..=2]).unwrap();
   /// assert!(t2.shape().iter().all(|x| *x==2));
   /// assert!(t2[&vec![1,1]] == 12.0 && t2[&vec![1,0]] == 11.0);
   /// 
   /// ```
   pub fn subtensor(&self, coords: &Vec<RangeInclusive<usize>>) -> Result<Self,String> {

      // Check boundaries
      assert_eq!(coords.len() , self.shape.len(), "All dimensions must be specified in the coordinate ranges.");
      assert!( coords.iter().zip(self.shape.iter()).all(|(rnge,dim)| rnge.start() >= &0 && rnge.end() <= &(dim-1) ), "Ranges for all dimensions must be within shape boundaries." );


      fn nested_subtensor<T: Clone>( t: &Tensor<T>,vrnge: &Vec<RangeInclusive<usize>>, depth: usize, acc_offset: usize) -> Vec<T> {         
         let rnge = vrnge[depth].clone();
         if depth < t.shape.len()-1 {
               rnge.clone().into_iter().map( |idx| nested_subtensor(t,vrnge, depth + 1 , acc_offset + idx.clone() * t.strides[depth])).flatten().collect::<Vec<T>>()
            } else {
               rnge.clone().into_iter().map(|dim_3| (t.data[dim_3 + acc_offset]).clone() ).collect::<Vec<T>>()
            }
      }

      let depth = 0;
      let acc_offset:usize = 0;

      // Setup new shape
      let shape = coords.iter().filter(|rnge| rnge.size_hint() != (1,Some(1)) ).map(|rnge| rnge.size_hint().0 ).collect::<Vec<_>>();

      Ok(Tensor::new(shape,
         &nested_subtensor(self,&coords,depth,acc_offset)))
   }

   pub fn zeros(dimensions: Vec<usize>) -> Self
   where
      T: Zero
   {
      let size = dimensions.iter().fold(1,|acc, x| acc * x);
      let zeros = vec![T::zero();size];

      Self::new(dimensions,&zeros)
   }

   pub fn ones(dimensions: Vec<usize>) -> Self
   where
      T: One
   {
      let size = dimensions.iter().fold(1,|acc, x| acc * x);
      let zeros = vec![T::one();size];

      Self::new(dimensions,&zeros)
   }

   pub fn fill(dimensions: Vec<usize>, val: T) -> Self
   where
      T: One
   {
      let size = dimensions.iter().fold(1,|acc, x| acc * x);
      let zeros = vec![val;size];

      Self::new(dimensions,&zeros)
   }

   pub fn arange(dimensions: Vec<usize>, val: T) -> Self
   where
      T: Float
   {
      let size = dimensions.iter().fold(1,|acc, x| acc * x);
      let arange:Vec<T> = (0..size)
         .enumerate()
         .map(|(_,idx)| val + T::from(idx).unwrap()  )
         .collect();

      Self::new(dimensions,&arange)
   }   

   pub fn random(dimensions: Vec<usize>, rnd_range: Range<T>) -> Self
   where
      T: SampleUniform + PartialOrd
   {
      let mut rng = rand::thread_rng(); // Create a thread-local RNG
      let size = dimensions.iter().fold(1,|acc, x| acc * x);
   
      // Fill the vector with random numbers in defined range
      let random_numbers:Vec<T> = (0..size)
         .map(|_| rng.gen_range(rnd_range.clone()) )
         .collect();

      Self::new(dimensions,&random_numbers)
   }

   pub fn shape(&self) -> Vec<usize>
   {
      self.shape.clone()
   }
}

/// Variadic tensor creation macro
/// ```
/// use crum::tensor;
/// 
/// let t3 = tensor![[
///      [
///            [1.0, 2.0, 3.0],
///            [4.0, 5.0, 6.0]
///      ],
///      [
///            [7.0, 8.0, 9.0],
///            [10.0, 11.0, 12.0]
///      ]
///   ]];
///   let t12 = tensor![[[[[[[[[[[[[8,4,5]]]]]]]]]]]]];
///
/// /* Output
/// Tensor { shape: [2, 2, 3], strides: [6, 3, 1], data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0] }
/// Tensor { shape: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3], strides: [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1], data: [8, 4, 5] }
///*/
/// ```
#[macro_export]
macro_rules! tensor {
   // Match rows and columns   
   ( $x:expr ) => {
      {
         let (t,d):(Vec<_>,Vec<usize>) = tensor_macro::tensor_flatten!($x);
         crum::tensor::Tensor::new(d,&t)
      }  
   };   
}