use std::any::Any;
use std::fmt::{Debug, Display};
use std::ops::{Add, Div, Index, IndexMut, Mul, Range, RangeInclusive, Sub};
use std::process::Output;
use num_traits::{Float, NumCast, One, Signed, ToPrimitive, Zero};
use rand::distributions::uniform::SampleUniform;
use rand::Rng;
use crate::tensor;



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

/// Implement * of two tensors generic tensor
/// Tensor contraction along a single dimension.
/// ```
/// use crum::tensor;
/// 
/// let a = tensor!([
///                  [1.0, 2.0, 3.0], 
///                  [4.0, 5.0, 6.0]
///                  ]); // Shape: (2,3)
///
/// let b = tensor!([
/// [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],  // j = 0
/// [[9.0, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]],  // j = 1
/// [[17.0, 18.0], [19.0, 20.0], [21.0, 22.0], [23.0, 24.0]]  // j = 2
/// ]); // Shape: (3,4,2)
/// 
/// res = a * b;
/// assert!(res.shape)
/// 
/// ```
impl<T: Clone + Zero + Mul<Output = T> + Debug + Display> Mul for Tensor<T>
{
   type Output = Tensor<T>;

   fn mul(self, rhs: Tensor<T>) -> Self {

      assert!(self.shape.last() == rhs.shape.first());

      let common_dim = self.shape[self.shape.len()-1];
      let stride = rhs.strides[0];
      let rh:Vec<T> =   (0..stride).into_iter().map(|dim| rhs.data.iter().skip(dim).step_by(stride).map(|elem| elem.clone() ).collect::<Vec<T>>() ).flatten().collect();

      let c:Vec<T> = self.data.chunks(common_dim)
                                 .map( |lh| rh.chunks(common_dim)
                                    .map(|rh|  lh.iter().zip(rh.iter()).fold(T::zero(),|acc,(l,r)|  acc + l.clone() * r.clone()) )).flatten().collect();
      
      Tensor::new( self.shape.iter().take(self.shape.len() - 1).chain(rhs.shape.iter().skip(1)).cloned().collect(), &c)
   }
}

/// Implement display of a generic matrix; {}
impl<T: Clone + Display> Display for Tensor<T>
{
   fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {

      fn nested_fmt<T: Clone + Display>(f: &mut std::fmt::Formatter<'_>, t: &Tensor<T>, depth: usize, acc_offset: usize) {         
         let rnge = 0..t.shape[depth];
         let spacing = 3*(depth);
         
         if depth < t.shape.len()-1 {
            write!(f, "\n{:>spacing$}","[").expect("Not Written");
            let rnge_iter = rnge.clone().into_iter().enumerate(); //.for_each(|(idx,rnge_idx)|nested_fmt(f, t, depth + 1 , acc_offset + rnge_idx.clone() * t.strides[depth]) );
            for (idx,rnge_idx) in rnge_iter {
               nested_fmt(f, t, depth + 1 , acc_offset + rnge_idx.clone() * t.strides[depth]);
               if idx < t.shape[depth]-1 {write!(f, ",").expect("Not Written")} 
                           else {write!(f, "\n{:>spacing$}","]").expect("Not Written")};
            }
         } else {
            write!(f, "\n{:>spacing$}","[").expect("Not Written");
            rnge.clone().into_iter()
               .enumerate()
               .for_each(|(idx,dim)| if idx < t.shape[depth] -1 {write!(f, "{:8.4}," , t.data[dim + acc_offset]).expect("Not Written") }
                                                                                    else {write!(f, "{:8.4}" , t.data[dim + acc_offset]).expect("Not Written") } );
            write!(f, "{}"," ]").expect("Not Written");
         }
      }
      write!(f, "\nshape: {:?}",self.shape).expect("Not Written");
      write!(f, "\nstrides: {:?}",self.strides).expect("Not Written");
      nested_fmt(f, self, 0, 0);
      Ok(())
   }
}

/// Implement standard functions for generic tensors
impl<T: Clone> Tensor<T>
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

      Ok(Tensor::new(coords.iter().filter(|rnge| rnge.size_hint() != (1,Some(1)) ).map(|rnge| rnge.size_hint().0 ).collect::<Vec<_>>(),
         &nested_subtensor(self,&coords,0,0)))
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

pub fn contract<T>(lh_idx: usize, lh_t: &Tensor<T>,rh_idx: usize, rh_t: &Tensor<T> ) -> Tensor<T>
where
   T: Clone + Zero + Mul<Output = T> + Debug + Display
{

   fn dim_right_shift<T: Clone>(new_dim:usize,old_dim:usize,data: &Vec<T>) -> Vec<T> {
      (0..new_dim).into_iter()
                                    .map( |dim|  data.chunks(old_dim)
                                    .skip(dim)
                                    .step_by(new_dim)
                                    .map(|chk| chk.to_vec()).flatten().collect::<Vec<_>>())
                                    .flatten()
                                    .collect()
   }


   assert!(lh_t.shape[lh_idx] == rh_t.shape[rh_idx]);

   let common_dim = lh_t.shape[lh_idx];

   let lstride = lh_t.strides[lh_idx];
   let rstride = rh_t.strides[rh_idx];

   let lh:Vec<T> =   (0..lstride).into_iter().map(|dim| lh_t.data.iter().skip(dim).step_by(lstride).map(|elem| elem.clone() ).collect::<Vec<T>>() ).flatten().collect();
   let mut rh:Vec<T> = (0..rstride).into_iter().map(|dim| rh_t.data.iter().skip(dim).step_by(rstride).map(|elem| elem.clone() ).collect::<Vec<T>>() ).flatten().collect::<Vec<_>>();

   // Shift both dimensions into correct positions
   // resample rh -> dimension right shift
   let mut shift_counter = rh_t.shape.len() - rh_idx - 1;
   while shift_counter > 0 {
      rh = dim_right_shift(rh_t.shape[shift_counter-1], rh_t.shape[shift_counter], &rh);
      shift_counter -= 1;       
   }
   
   //println!("lh {:?}", lh);
   //println!("rh {:?}", rh); 
      
   let mut c:Vec<T> = lh.chunks(common_dim)
                     .map( |ls| rh.chunks(common_dim)
                     .map(|rs|  ls.iter().zip(rs.iter()).fold(T::zero(),|acc,(l,r)|  acc + l.clone() * r.clone()) )).flatten().collect();

   // resample c -> dimension
   shift_counter = lh_t.shape.len() - lh_idx - 1;
   while shift_counter > 0 {
      c = dim_right_shift(lh_t.shape[shift_counter-1], lh_t.shape[shift_counter], &c);
      shift_counter -= 1;       
   }
                  

   let mut lh_sh = lh_t.shape.clone();
   lh_sh.remove(lh_idx);
   let mut rh_sh = rh_t.shape.clone();
   rh_sh.remove(rh_idx);
   let newshape = lh_sh.iter().chain(rh_sh.iter()).cloned().collect();
      
   Tensor::new( newshape, &c)
   
   
   //Tensor::new( vec![1],&vec![T::zero()])
}

/// Einsten's Tensor contraction notation implementation
/// Should be a macro to consume string and operands ...
/// # Arguments
///
/// * `equation` - Einstein's summation notation for tensor contraction.
/// * `operands` - n-tuple of tensors.
///
/// # Returns
///
/// Matrices L(Lower Triangular),U(Upper Triangular),P(Permutation) and the number of rows swaps in the pivot process.
///
pub fn einsum<T,U>( equation: String, operands: U) -> Tensor<T> {

   // Check if '->' and split equation on that.
   let eqn_parts = equation.split("->");
   if eqn_parts.count() == 1 {
      // no rhs => straight contraction preserving index order


   


   } else {
      // resultant ordering required



   }
   
   for &byte in equation.as_bytes() {
      
      
      
      
      println!("{}", byte as char); // Convert byte back to char
   }






   todo!()
}
   