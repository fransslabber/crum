use std::ops::{Add, Div, Index, IndexMut, Mul, Range, RangeInclusive, Sub};
use num_traits::{Float, NumCast, One, Signed, ToPrimitive, Zero};
use rand::distributions::uniform::SampleUniform;
use rand::Rng;

pub type TCoord = Vec<usize>;

pub struct TSeq {
   first: usize,
   last: usize,
   step: usize
}

// const fn square(x: i32) -> i32 {
//    x * x
// // }


//    constexpr FASTOR_INLINE seq(int _f, int _l, int _s=1) : _first(_f), _last(_l), _step(_s) {}
//    constexpr FASTOR_INLINE seq(int num) : _first(num), _last(num+1), _step(1) {}

//    template<int F, int L, int S=1>
//    constexpr FASTOR_INLINE seq(fseq<F,L,S>) : _first(F), _last(L), _step(S) {}

//    // Do not allow construction of seq using std::initializer_list, as it happens
//    // implicitly. Overloading operator() with std::initializer_list should imply
//    // TensorRandomView, not TensorView
//    template<typename T>
//    constexpr FASTOR_INLINE seq(std::initializer_list<T> _s1) = delete;

//    // Do not provide this overload as it is meaningless [iseq stands for immediate evaluation]
//    // template<size_t F, size_t L, size_t S=1>
//    // constexpr FASTOR_INLINE seq(iseq<F,L,S>) : _first(F), _last(L), _step(S) {}

//    FASTOR_INLINE int size() const {
//        int range = _last - _first;
//        return range % _step==0 ? range/_step : range/_step+1;
//    }

//    constexpr FASTOR_INLINE bool operator==(seq other) const {
//        return (_first==other._first && _last==other._last && _step==other._step) ? true : false;
//    }
//    constexpr FASTOR_INLINE bool operator!=(seq other) const {
//        return (_first!=other._first || _last!=other._last || _step!=other._step) ? true : false;
//    }


// structure with data stored as row dominant
#[derive(Clone,Debug)]
pub struct Tensor<T> 
{
   dim: Vec<usize>,
   strides: Vec<usize>,
   data: Vec<T>
}

/// Implement indexing of a generic tensor
impl<T> Index<&TCoord> for Tensor<T>
{
   type Output = T;

   fn index(&self, coord: &TCoord) -> &T {
      &self.data[coord.iter().zip(self.strides.iter()).fold(0, |acc,(a,b)| acc + a * b )]
   }
}

/// Implement mutable indexing of a generic matrix; matrix[(i,j,)] = a
impl<T> IndexMut<&TCoord> for Tensor<T>
{
   fn index_mut(&mut self, coord: &TCoord) -> &mut Self::Output {
      &mut self.data[coord.iter().zip(self.strides.iter()).fold(0, |acc,(a,b)| acc + a * b )]
   }
}


/// Implement standard functions for generic tensors
impl<T: Clone + Copy> Tensor<T>
{
   /// Constructor for a new matrix from Vec
   pub fn new(dimensions: TCoord, data: &Vec<T> ) -> Self
   {
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

   pub fn zeros(dimensions: TCoord) -> Self
   where
      T: Zero
   {
      let size = dimensions.iter().fold(1,|acc, x| acc * x);
      let zeros = vec![T::zero();size];

      Self::new(dimensions,&zeros)
   }

   pub fn ones(dimensions: TCoord) -> Self
   where
      T: One
   {
      let size = dimensions.iter().fold(1,|acc, x| acc * x);
      let zeros = vec![T::one();size];

      Self::new(dimensions,&zeros)
   }

   pub fn fill(dimensions: TCoord, val: T) -> Self
   where
      T: One
   {
      let size = dimensions.iter().fold(1,|acc, x| acc * x);
      let zeros = vec![val;size];

      Self::new(dimensions,&zeros)
   }

   pub fn arange(dimensions: TCoord, val: T) -> Self
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

   pub fn random(dimensions: TCoord, rnd_range: Range<T>) -> Self
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

}
