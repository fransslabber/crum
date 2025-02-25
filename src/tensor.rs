use std::fmt::{Debug, Display};
use std::ops::{Index, IndexMut, Mul, Range, RangeInclusive};
use num_traits::{Float, One, Zero};
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

   pub fn reshape(&self, shape: Vec<usize> )  -> Self
   {
      Tensor::new(shape, &self.data)
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

   pub fn tensordot(&self,rh: &Tensor<T>, axes: Vec<Vec<usize>>) -> Self 
   where
      T: Copy + Default + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Zero
   {

      let a_contract = &axes[0];
      let b_contract = &axes[1];
      let mut a_free = Vec::<usize>::new();
      let mut b_free = Vec::<usize>::new();
      
      for a_axis in 0..self.shape.len(){
         if !a_contract.contains(&a_axis) {
            a_free.push(a_axis);
         }
      }
      for b_axis in 0..rh.shape.len() {
         if !b_contract.contains(&b_axis) {
            b_free.push(b_axis);
         }
      }

      self.contract(rh,&a_contract, &b_contract, &a_free, &b_free)
   }

   ///
   /// Einstein's Summation Notation Contraction
   /// 
   ///
   /// #Example
   /// ```
   /// use crum::tensor::Tensor;
   /// let a = Tensor::arange(vec![2, 3, 4, 5], 1.0);
   /// let b = Tensor::arange(vec![5, 4, 3, 2], 1.0);
   /// let c = a.einsum(&b, "ijkl,mkni");
   /// 
   /// assert!(c.shape() == vec![3,5,5,3]);
   /// assert!(c[&vec![2,4,4,2]] == 73350.0);
   /// 
   /// ```
   pub fn einsum( &self, rht: &Tensor<T>, eqn: &str ) -> Tensor<T>
   where
      T: Copy + Default + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Zero,
  {
      // Parse contraction equation
      let parts: Vec<&str> = eqn.split("->").collect();
      let input_part = parts[0];
      let _output_part = if parts.len() > 1 { parts[1] } else { "" };
      let inputs: Vec<&str> = input_part.split(',').collect();

      let mut a_contract = Vec::<usize>::new();
      let mut b_contract = Vec::<usize>::new();
      let mut a_free = Vec::<usize>::new();
      let mut b_free = Vec::<usize>::new();

      for (lh_idx,lh_char) in inputs[0].char_indices() {
         for (rh_idx,rh_char) in inputs[1].char_indices() {
            if lh_char == rh_char {
               a_contract.push(lh_idx);
               b_contract.push(rh_idx); 
            }
         }
      }
      for (lh_idx,_) in inputs[0].char_indices() {
         if !a_contract.contains(&lh_idx) {
            a_free.push(lh_idx);
         }
      }
      for (rh_idx,_) in inputs[1].char_indices() {
         if !b_contract.contains(&rh_idx) {
            b_free.push(rh_idx);
         }
      }
      self.contract(rht,&a_contract, &b_contract, &a_free, &b_free)
   } 

   fn contract(&self, rh: &Tensor<T>, a_contract: &Vec<usize> , b_contract: &Vec<usize> , a_free: &Vec<usize> , b_free: &Vec<usize> ) -> Self
   where
      T: Copy + Default + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Zero
   {
      // println!("a_free {:?}",a_free);
      // println!("a_contract {:?}",a_contract);
      // println!("b_contract {:?}",b_contract);
      // println!("b_free {:?}",b_free);
      
      assert!( a_contract.iter().zip(b_contract.iter()).all(|(a,b)| self.shape[*a] == rh.shape[*b] ) );

      // Permute tensors
      let a_full:Vec<usize> = a_free.iter().chain(a_contract.iter()).map(|x| *x).collect();
      //println!("a_full {:?}",a_full);
      let a_trans = self.transpose(&a_full); // Move contract_a to end
      let b_full:Vec<usize> = b_contract.iter().chain(b_free.iter()).map(|x| *x).collect();
      //println!("b_full {:?}",b_full);
      let b_trans = rh.transpose(&b_full); // Move contract_b to start
      
      // Compute sizes
      let free_a_size = a_free.iter().map(|&i| self.shape[i]).product::<usize>();
      let contract_size = a_contract.iter().map(|&i| self.shape[i]).product::<usize>();
      let free_b_size = b_free.iter().map(|&i| rh.shape[i]).product::<usize>();
      
      // Reshape
      let a_reshaped = a_trans.reshape(vec![free_a_size, contract_size]);
      let b_reshaped = b_trans.reshape(vec![contract_size, free_b_size]);

      // println!("{}",a_reshaped);
      // println!("{}",b_reshaped);

      // Matrix multiplication
      let c = Tensor::matmul(&a_reshaped,&b_reshaped);
      
      // Reshape to final shape
      let mut c_shape = a_free.iter().map(|&i| self.shape[i]).collect::<Vec<_>>();
      c_shape.extend(b_free.iter().map(|&i| rh.shape[i]));
      c.reshape(c_shape)
   }

   /// Matrix multiplication
   pub fn matmul(lh: &Self,rh: &Self) -> Self 
   where T: Default + Zero + Mul<Output = T> {
 
      let rht = rh.transpose(&[1,0]);

      let c_data:Vec<T> = lh.data.chunks(lh.shape[1])
         .map( | row|
            // for each row chunk dot with every col chunk
            rht.data.chunks(rht.shape[1])
                     .map(|col|
                        col.iter().zip(row.iter()).fold(T::zero(),|acc,(l,r)|  acc + l.clone() * r.clone())
                     ).collect::<Vec<_>>()
                  ).flatten().collect();

      Tensor::new(vec![lh.shape[0],rh.shape[1]], &c_data)
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
   where T: Default + Clone{
      let mut ret = Tensor::new(order.iter().map(|&i| self.shape[i]).collect(),
         &vec![T::default(); self.data.len()] );

      let mut current_index = vec![0 as usize;self.shape.len()];
      
      fn permute<T: Clone>(t: &Tensor<T>, nt: &mut Tensor<T>, order: &[usize], current: &mut Vec<usize>, depth: usize  )
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
