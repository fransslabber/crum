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

#[derive(Clone,Debug)]
pub struct Cursor<'a, T> 
{
   depth: usize,
   accumulated_offset: usize,
   t: &'a Tensor<T>
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
/// Tensor contraction along a common index.
impl<T: Clone + Zero + Mul<Output = T> + Debug + Display> Mul for Tensor<T>
{
   type Output = Tensor<T>;

   fn mul(self, rhs: Tensor<T>) -> Self {

      assert!(self.shape.last() == rhs.shape.first());


      // fn rh_nested_single_row<T: Clone + Debug>( t: &Tensor<T>, depth: usize, acc_offset: usize, ) -> Vec<T> {         
      //    let rnge = 0..t.shape[depth];
      //    if depth < t.shape.len()-2 {
      //          rnge.clone().into_iter().map( |idx| rh_nested_single_row(t, depth + 1 , acc_offset + idx.clone() * t.strides[depth])).flatten().collect::<Vec<T>>()
      //       } else {
      //          //println!("rnge {:?}",rnge);
      //          let single_row = (0..t.shape[depth]).map(|final_dim|    
               
      //                         rnge.clone().into_iter()
      //                         .map(|dim_n_1|  (0..t.shape[depth+1]).clone().into_iter().skip(final_dim).step_by(t.shape[depth]).map(|dim_n| (t.data[dim_n + acc_offset + dim_n_1 * t.strides[depth]]).clone() ).collect::<Vec<T>>() ).flatten().collect::<Vec<T>>()
      //          ).flatten().collect();
      //          println!("rhs {:?}",single_row);
      //          single_row
      //       }
      // }
      // let rh = rh_nested_single_row(&rhs,0,0);
      let common_dim = self.shape[self.shape.len()-1];
      let stride = rhs.strides[0];
      let rh:Vec<T> =   (0..stride).into_iter().map(|dim| rhs.data.iter().skip(dim).step_by(stride).map(|elem| elem.clone() ).collect::<Vec<T>>() ).flatten().collect();

      println!("rh {:?}", rh);
      // LHS initial ranges
      //let lhs_len = self.shape.len();
      //let mut lhs_ranges:Vec<RangeInclusive<usize>> = self.shape.iter().take(lhs_len - 1).map(|_| (0..=0) ).collect();
      //lhs_ranges.push(0..=*self.shape.last().unwrap());
      //let mut lhs_index = vec![0,lhs_len];
      // fn lh_nested_single_row<T: Clone + Debug>( t: &Tensor<T>, depth: usize, acc_offset: usize) -> Vec<T> {         
      //    let rnge = 0..t.shape[depth];
      //    if depth < t.shape.len()-1 {
      //          rnge.clone().into_iter().map( |idx| lh_nested_single_row(t, depth + 1 , acc_offset + idx.clone() * t.strides[depth])).flatten().collect::<Vec<T>>()
      //       } else {
      //          let l_single_row = rnge.clone().into_iter().map(|dim| (t.data[dim + acc_offset]).clone()).collect::<Vec<T>>();
      //          println!("lhs {:?}",l_single_row);
      //          l_single_row
      //       }
      // }
      // let lh = lh_nested_single_row(&self,0,0);
      // println!("{:?}",lh);

      
      let c:Vec<T> = self.data.chunks(common_dim)
                                 .map( |lh| rh.chunks(common_dim)
                                    .map(|rh|  lh.iter().zip(rh.iter()).inspect(|(l,r)| println!("{} * {}",l,r)).fold(T::zero(),|acc,(l,r)|  acc + l.clone() * r.clone()) )).flatten().collect();
      //println!("{:?}",c);
      //let new_elem = rh.chunks(2).zip(rnge.clone().into_iter()).

      
      // RHS initial ranges
      // let rhs_len = rhs.shape.len();
      // let mut rhs_ranges:Vec<RangeInclusive<usize>> = rhs.shape.iter().take(rhs_len - 2).map(|_| (0..=0) ).collect();
      // rhs_ranges.push(0..=rhs.shape[rhs_len-2]);
      // rhs_ranges.push(0..=0);
      // println!("rhs_ranges {:?}",rhs_ranges);

      // // Combining 
      // fn lh_nested_mul<T: Clone>( t: &Tensor<T>,vrnge: &Vec<RangeInclusive<usize>>, depth: usize, acc_offset: usize) -> Vec<T> {         
      //    let rnge = vrnge[depth].clone();
      //    if depth < t.shape.len()-1 {
      //          rnge.clone().into_iter().map( |idx| nested_subtensor(t,vrnge, depth + 1 , acc_offset + idx.clone() * t.strides[depth])).flatten().collect::<Vec<T>>()
      //       } else {
      //          rnge.clone().into_iter().map(|dim| (t.data[dim + acc_offset]).clone() ).collect::<Vec<T>>()
      //       }
      // }


      //let rhs_len = rhs.shape.len();
      //let rhs_coords:Vec<usize>= vec![0;rhs_len];
      //let mut rhs_rnge:Vec<RangeInclusive<usize>> = rhs_coords.iter().map(|_| (0..=0) ).collect();
      //lhs_rnge[rhs_len-1] = 0..=self.shape[self.shape.len()-1]-1;
      //println!("{:?}",lhs_rnge);
      //println!("{:?}", rhs.subtensor(&vec![0..=0,0..=1,0..=0]));


      // let lhs_step:Vec<usize>= vec![0;self.shape.len()];
      // let rhs_step:Vec<usize>= vec![0;rhs.shape.len()];

      // // Compile inclusive ranges for all dimensions in both lhs and rhs tensors
      // let mut lhs_rnge:Vec<RangeInclusive<usize>> = self.shape.iter().map(|r| 0..=(*r-1) ).collect();
      // let lhs_len = lhs_rnge.len();
      // let mut rhs_rnge:Vec<RangeInclusive<usize>> = rhs.shape.iter().map(|r|  0..=(*r-1) ).collect();
      // let rhs_len = rhs_rnge.len();

      // fn nested_subtensor<T: Clone>( t: &Tensor<T>, depth: usize, acc_offset: usize) -> Vec<T> {         
      //    let rnge = 0..t.shape[depth];
      //    if depth < t.shape.len()-1 {
      //          rnge.clone().into_iter().map( |idx| nested_subtensor(t, depth + 1 , acc_offset + idx.clone() * t.strides[depth])).flatten().collect::<Vec<T>>()
      //       } else {
      //          rnge.clone().into_iter().map(|dim| (t.data[dim + acc_offset]).clone() ).collect::<Vec<T>>()
      //       }
      // }
      
      //for dim in self.shape {
         




      //}

      // let mut prod_vec = Vec::<T>::new();
      // fn nested_mul<T: Clone>( c1: &Cursor<T>, c2: &Cursor<T>, ) -> Vec<T> {         
      //    let rnge = 0..c1.t.shape[c1.depth];
      //    if c1.depth < c1.t.shape.len()-1 {
      //          rnge.clone().into_iter().map( |idx| nested_mul( &Cursor { depth: c1.depth + 1 , accumulated_offset: c1.accumulated_offset + idx.clone() * c1.t.strides[c1.depth], t: c1.t}, c2 ))
      //                                  .flatten().collect::<Vec<T>>()
      //       } else {
      //          let lhs = rnge.clone().into_iter().map(|dim_3| (c1.t.data[dim_3 + c1.accumulated_offset]).clone() ).collect::<Vec<T>>();
      //          let rhs = nested_mul(c2,c1);

      //          let product = lhs_vec.iter().zip(rhs_vec.iter()).fold(T::zero(),|acc,(x,y)| acc + x.clone() * y.clone());

      //          prod_vec.push(product);     

      //       }
      // }

      // fn nested_subtensor<T: Clone>( t: &Tensor<T>,vrnge: &Vec<RangeInclusive<usize>>, depth: usize, acc_offset: usize) -> Vec<T> {         
      //    let rnge = vrnge[depth].clone();
      //    if depth < t.shape.len()-1 {
      //          rnge.clone().into_iter().map( |idx| nested_subtensor(t,vrnge, depth + 1 , acc_offset + idx.clone() * t.strides[depth])).flatten().collect::<Vec<T>>()
      //       } else {
      //          rnge.clone().into_iter().map(|dim| (t.data[dim + acc_offset]).clone() ).collect::<Vec<T>>()
      //       }
      // }
      // // let result_size = self.shape.iter().take(self.shape.len() - 1).fold(0,|acc,x| acc * *x) 
      // //                      * rhs.shape.iter().skip(1).fold(0,|acc,x| acc * *x);
      // let mut prod_vec = Vec::<T>::new();
      // for combined_idx in 0..*self.shape.last().unwrap() { 
      //    lhs_rnge[lhs_len-1] = combined_idx..=combined_idx;
      //    println!("lhs_rnge {:?}", lhs_rnge);
      //    let lhs_vec = nested_subtensor(&self,&lhs_rnge, 0,0); 
      //    rhs_rnge[0] = combined_idx..=combined_idx;
      //    let rhs_vec = nested_subtensor(&rhs,&rhs_rnge, 0,0);
      //    println!("{combined_idx} -> lhs: {:?} rhs {:?}", lhs_vec, rhs_vec);
      //    let product = lhs_vec.iter().zip(rhs_vec.iter()).fold(T::zero(),|acc,(x,y)| acc + x.clone() * y.clone());

      //    prod_vec.push(product);         
      // }

      let new_shape:Vec<usize> = self.shape.iter().take(self.shape.len() - 1).chain(rhs.shape.iter().skip(1)).cloned().collect();
      //println!("{:?}", new_shape);

      
      Tensor::new( new_shape, &c)


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