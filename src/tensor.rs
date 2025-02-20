use std::any::Any;
use std::fmt::{Debug, Display};
use std::ops::{Add, AddAssign, Div, Index, IndexMut, Mul, Range, RangeInclusive, Sub};
use std::process::Output;
use num_traits::{Float, NumCast, One, Signed, ToPrimitive, Zero};
use rand::distributions::uniform::SampleUniform;
use rand::Rng;
use crate::tensor;
use std::collections::HashMap;



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

impl<T: PartialEq> PartialEq for Tensor<T>
{
   fn eq(&self, other: &Self) -> bool {
      self.shape == other.shape && self.strides == other.strides && self.data == other.data
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

   pub fn data(&self) -> Vec<T>
   {
      self.data.clone()
   }

   ///
   /// Contraction
   /// 







   fn compute_strides(shape: &[usize]) -> Vec<usize> {
      let mut strides = vec![1; shape.len()];
      for i in (0..shape.len() - 1).rev() {
          strides[i] = strides[i + 1] * shape[i + 1];
      }
      strides
  }


   pub fn contract(&self, other: &Self, einsum: &str) -> Self
   where T: Mul<Output = T> + AddAssign + Default {
      let (a_indices, b_indices, c_indices) = Self::parse_einsum(einsum);
      let contracted_indices: Vec<char> = a_indices.iter()
         .filter(|c| b_indices.contains(c))
         .cloned()
         .collect();

      // Validate contracted dimensions match
      for &idx in &contracted_indices {
         let a_pos = a_indices.iter().position(|&c| c == idx).unwrap();
         let b_pos = b_indices.iter().position(|&c| c == idx).unwrap();
         assert_eq!(self.shape[a_pos], other.shape[b_pos], "Contracted dimension mismatch for index {}", idx);
      }

      // Compute output shape
      let mut c_shape = Vec::new();
      for &idx in &c_indices {
         if a_indices.contains(&idx) {
            let pos = a_indices.iter().position(|&c| c == idx).unwrap();
            c_shape.push(self.shape[pos]);
         } else {
            let pos = b_indices.iter().position(|&c| c == idx).unwrap();
            c_shape.push(other.shape[pos]);
         }
      }

      // Precompute index mappings for A and B
      let a_map: HashMap<_, _> = a_indices.iter().enumerate()
         .map(|(i, &c)| (c, (i, self.strides[i])))
         .collect();
      let b_map: HashMap<_, _> = b_indices.iter().enumerate()
         .map(|(i, &c)| (c, (i, other.strides[i])))
         .collect();

      // Compute strides for output tensor
      let c_strides = Self::compute_strides(&c_shape);
      let mut result_data = vec![T::default(); c_shape.iter().product()];

      // Iterate over each element in the output tensor
      for c_flat in 0..result_data.len() {
          // Convert flat index to multi-dimensional indices for output
         let c_indices_values = Self::flat_to_indices(c_flat, &c_strides, &c_shape);

          // Calculate base offsets in A and B for non-contracted indices
         let (a_base, b_base) = self.calculate_bases(&c_indices, &c_indices_values, &a_map, &b_map);

          // Calculate sum over contracted indices
         let sum = self.calculate_sum(a_base, b_base, &contracted_indices, &a_map, &b_map);

         result_data[c_flat] = sum;
      }

      Tensor {
         shape: c_shape,
         strides: c_strides,
         data: result_data,
      }
  }

   pub fn parse_einsum(einsum: &str) -> (Vec<char>, Vec<char>, Vec<char>) {
         let parts: Vec<&str> = einsum.split("->").collect();
         let input_part = parts[0];
         let output_part = if parts.len() > 1 { parts[1] } else { "" };

         let inputs: Vec<&str> = input_part.split(',').collect();
         let a_indices: Vec<char> = inputs[0].chars().collect();
         let b_indices: Vec<char> = inputs[1].chars().collect();

         // Infer output indices if not explicitly provided
         let c_indices = if output_part.is_empty() {
            let mut unique_indices = Vec::new();
            for &c in &a_indices {
               if !b_indices.contains(&c) {
                     unique_indices.push(c);
               }
            }
            for &c in &b_indices {
               if !a_indices.contains(&c) && !unique_indices.contains(&c) {
                     unique_indices.push(c);
               }
            }
            unique_indices
         } else {
            output_part.chars().collect()
         };

         (a_indices, b_indices, c_indices)
   }

  fn flat_to_indices(flat: usize, strides: &[usize], shape: &[usize]) -> Vec<usize> {
      let mut indices = vec![0; shape.len()];
      let mut remaining = flat;
      for (i, &stride) in strides.iter().enumerate() {
          indices[i] = remaining / stride;
          remaining %= stride;
      }
      indices
  }

  fn calculate_bases(
      &self,
      c_indices: &[char],
      c_indices_values: &[usize],
      a_map: &HashMap<char, (usize, usize)>,
      b_map: &HashMap<char, (usize, usize)>,
  ) -> (usize, usize) {
      let mut a_base = 0;
      let mut b_base = 0;
      for (i, &idx) in c_indices.iter().enumerate() {
          if let Some(&(pos, stride)) = a_map.get(&idx) {
              a_base += c_indices_values[i] * stride;
          }
          if let Some(&(pos, stride)) = b_map.get(&idx) {
              b_base += c_indices_values[i] * stride;
          }
      }
      (a_base, b_base)
  }

  fn calculate_sum(
      &self,
      a_base: usize,
      b_base: usize,
      contracted_indices: &[char],
      a_map: &HashMap<char, (usize, usize)>,
      b_map: &HashMap<char, (usize, usize)>,
  ) -> T
  where T: Mul<Output = T> + AddAssign + Default{
      let contracted_shapes: Vec<usize> = contracted_indices.iter()
          .map(|&c| {
              let (pos, _) = a_map[&c];
              self.shape[pos]
          })
          .collect();
      let total_contracted: usize = contracted_shapes.iter().product();
      let mut sum = T::default();

      for flat in 0..total_contracted {
          let contracted_values = Self::flat_to_indices(flat, &Self::compute_strides(&contracted_shapes), &contracted_shapes);
          let mut a_offset = a_base;
          let mut b_offset = b_base;
          for (i, &idx) in contracted_indices.iter().enumerate() {
              let value = contracted_values[i];
              let (_, a_stride) = a_map[&idx];
              a_offset += value * a_stride;
              let (_, b_stride) = b_map[&idx];
              b_offset += value * b_stride;
          }
          sum += self.data[a_offset].clone() * self.data[b_offset].clone();
      }

      sum
  }

   /// Permutes(transposes) the dimensions of the tensor according to the given order.
   /// ```
   /// use crum::tensor::Tensor;
   /// use crum::tensor;
   /// 
   /// let a = Tensor::arange(vec![2, 2, 2], 1.0);
   /// let at = a.transpose(&[2,1,0]);
   /// let compare = tensor!([[[1.0, 5.0],[3.0, 7.0]],
   ///                         [[2.0, 6.0],[4.0, 8.0]]]);
   /// assert!(at == compare);
   /// ```
   pub fn transpose(&self, order: &[usize]) -> Self
   where T: Default + Clone + Debug + Display{
      let mut ret = Tensor::new(order.iter().map(|&i| self.shape[i]).collect(),
         &vec![T::default(); self.data.len()] );

      let mut current_index = vec![0 as usize;self.shape.len()];
      
      fn permute<T: Clone + Debug + Display>(t: &Tensor<T>, nt: &mut Tensor<T>, order: &[usize], current: &mut Vec<usize>, depth: usize  )
      { 
         if depth < t.shape.len() {
            (0..t.shape[depth]).into_iter().for_each(|idx| {current[depth] = idx;  permute(t, nt, order, current, depth+1 )})
         } else {

            let old_offset = t.strides.iter().zip(current.iter()).fold(0,|acc,(x,y)|   acc + x * y);
            let new_coord:Vec<usize> = order.iter().map(|c| current[*c] ).collect();
            let new_offset = nt.strides.iter().zip(new_coord.iter()).fold(0,|acc,(x,y)|   acc + x * y);
            nt.data[new_offset] = t.data[old_offset].clone();
         }
      }
      permute(self, &mut ret,order, &mut current_index, 0);
      ret      
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

pub fn flatslice<T: Clone + Debug + Display>(t: &Tensor<T>, target_dim: usize, depth: usize, acc_offset: usize ) -> Vec<T> {      
   let rnge = 0..t.shape[depth];   
   if depth < target_dim {
      //println!("deeper: depth {depth} {:?}", rnge);
      rnge.into_iter().map( |dim_count| flatslice(t,target_dim, depth + 1,acc_offset + (dim_count.clone() * t.strides[depth]))).flatten().collect::<Vec<T>>()
   } else {
      //println!("return:  depth {depth} stride {} trgt dim {target_dim} offset {acc_offset} {:?}",t.strides[depth], rnge );
      let stride = t.strides[depth];
      let ret = (0..stride).into_iter().map(|idx|                                          
                                             t.data.iter()                                 
                                             .skip(acc_offset + idx)
                                             .step_by(stride)
                                             .take(t.shape[depth])
                                             .map(|x| x.clone() ).collect::<Vec<T>>()
                                             ).flatten().collect::<Vec<_>>();
      
      //println!("return: fragment {:?}",ret);
      ret
   }   
}

pub fn dotslice<T: Clone + Zero + Mul<Output = T> + Debug>(lh: &Vec<T>, rh: &Vec<T>,common_dim: Vec<usize> ) -> Vec<T> {
      
      let rh_skip = vec![0,1];
      let rh_step = vec![1,2];

      // rearrange rh?
      let mut rhext = rh.clone();
      rhext.swap(2, 4);
      rhext.swap(3, 5);


      lh.chunks(4)
         //.inspect(|r| println!("lchunk {:?}",r))
         .map( | ls|

            // for each lh chunk dot with 
            rhext.chunks(4)
            //.inspect(|r| println!("rchunk {:?} ",r))
            .map(|rs|  ls.iter()
                                 .zip(rs.iter())
                                 .fold(T::zero(),|acc,(l,r)|  acc + l.clone() * r.clone())
            )
            //.inspect(|val| println!("{val}"))
         )
         .flatten()
         .collect::<Vec<_>>()  



} 

// pub fn contract<T>(lh_idx: Vec<usize>, lh_t: &Tensor<T>,rh_idx: Vec<usize>, rh_t: &Tensor<T> ) -> Tensor<T>
// where
//    T: Clone + Zero + Mul<Output = T> + Debug + Display + One
// {
//    assert!(lh_idx.iter().zip(rh_idx.iter().rev()).all(|(&l,&r)| lh_t.shape[l] == rh_t.shape[r]) );
//   // Build new shape
//    let lh_sh:Vec<usize> = lh_t.shape.iter().enumerate().filter(|(offset,_)| lh_idx.contains(offset) == false ).map(|(_,x)| *x).collect();
//    //let lh_cumulative_dim = lh_sh.iter().fold(1 as usize,|acc,x| acc * *x);
//    //println!("lh_sh {:?} {lh_cumulative_dim}",lh_sh);
//    let rh_sh:Vec<usize> = rh_t.shape.iter().enumerate().filter(|(offset,_)| rh_idx.contains(offset) == false ).map(|(_,x): (usize, &usize)| *x).collect();
//    //let rh_cumulative_dim = rh_sh.iter().fold(1 as usize,|acc,x| acc * *x);
//    //println!("rh_sh {:?} {rh_cumulative_dim}",rh_sh);
//    let new_shape = lh_sh.iter().chain(rh_sh.iter()).cloned().collect();
//    //let initial_contract_dim_size = lh_t.shape[*lh_idx.last().unwrap()];

//    // Get flatpacked tensors for both sides with contracting dimension  
//    let lh = flatslice(lh_t,*lh_idx.last().unwrap(),0,0);  
//    let rh = flatslice(rh_t,*rh_idx.first().unwrap(),0,0);
//    println!("lh {:?} {}", lh, lh.len());
//    println!("rh {:?} {}", rh, rh.len());

//    let c = dotslice(&lh, &rh, lh_idx.iter().map(|idx| lh_t.shape[*idx]).collect());
//    //println!("{} {:?}",c.len(), c);

//    Tensor::new( new_shape, &c)
//    //Tensor::new( vec![1],&vec![T::zero()])
// }

