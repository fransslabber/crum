use crate::complex::Complex;
use crate::matrix;

use std::ops::{Add, Div, Index, IndexMut, Mul, RangeInclusive, Sub,};
use std::slice::IterMut;
use std::vec::Vec;
use std::fmt::{Debug, Display};
use num_traits::float::FloatCore;
use num_traits::{Float, One, Zero,Signed};
use rand::distributions::uniform::SampleUniform;
use rand::Rng;
use std::iter::{Enumerate, Skip};

///
/// Some linear algebra operations on vector quantities
/// All operations are generic on floats and integers
///

/// Compute the dot product of two vectors and returns the result.
///
/// # Arguments
///
/// * `v1` - The first vector.
/// * `v2` - The second vector.
///
/// # Returns
///
/// The dot product of `v1` and `v2`.
///
/// # Example
///
/// ```
/// use crum::matrix::dot_product;
/// use crum::complex::Complex;
/// let result = dot_product(&vec![5, 3],&vec![6,2]);
/// assert_eq!(result, 36);
/// ```
pub fn dot_product<T>(v1: &Vec<T>, v2: &Vec<T>) -> T
where
   T: Copy + Zero + Mul<Output = T>
{
   v1.iter().zip(v2).fold(T::zero(), |acc, (&x, &y)| acc + x * y)
}

/// Compute the magnitude of a vector
#[allow(dead_code)]
pub fn magnitude<T>(v: &Vec<T>) -> T
where
   T: Copy + Float
{
   let sum = v.iter().fold(T::zero(), |acc, &x| acc + x * x);
   sum.sqrt()
}

/// Compute the scalar * vector
#[allow(dead_code)]
fn scalar_mul<T>(v: &Vec<T>, a:T) -> Vec<T>
where
   T: Clone + Mul<Output = T>
{
   v.iter().map(|x| a.clone() * x.clone()).collect()
}

/// Compute col vector * row vector = square matrix
#[allow(dead_code)]
fn cvec_rvec<T>(v1: &Vec<T>, v2: &Vec<T>) -> Matrix<T> 
where 
   T: Clone + Zero + Float
{
   assert_eq!(v1.len(),v2.len(), "Vectors must have same dimensions.");

   let mut data = vec![T::zero() ; v1.len()*v1.len()];
   for (i,colvec) in v1.iter().enumerate() {
      for (j,rowvec) in v2.iter().enumerate() {
         data[(v1.len()*i) + j ] = colvec.clone() * rowvec.clone();
      }

   }
   Matrix::<T>::new(v1.len() as u128,v1.len() as u128,data)
}

/// Compute the vector / scalar
#[allow(dead_code)]
fn scalar_div<T>(v: &Vec<T>, a:T) -> Vec<T>
where
   T: Clone + Div<Output = T>
{
   v.iter().map(|x| x.clone()/a.clone()).collect()
}

/// Compute vector - vector
#[allow(dead_code)]
fn vector_sub<T>(v1: &Vec<T>, v2: &Vec<T>) -> Vec<T>
where
   T: Copy + Sub<Output = T>
{
   v1.iter().zip(v2).map(|(&x,&y)| x - y).collect()
}

/// Compute vector + vector
#[allow(dead_code)]
fn vector_add<T>(v1: &Vec<T>, v2: &Vec<T>) -> Vec<T>
where
   T: Copy + Add<Output = T>
{
   v1.iter().zip(v2).map(|(&x,&y)| x + y).collect()
}

/// Construct an nth identity vector [ 0 , 0, ... , 1, ... ,0 ]
fn nth_identity_vector<T>(idx: usize, size: usize) -> Vec<T>
where
   T: Copy + Zero + One
{
   let mut vec = vec![T::zero(); size];
   vec[idx-1] = T::one();
   vec
}

/// Define a generic matrix structure with data stored as row dominant
#[derive(Clone)]
pub struct Matrix<T> 
{
   rows: u128,
   cols: u128,
   data: Vec<T>
}

/// Implement the + trait for a generic matrix
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

/// Implement the - trait for a generic matrix
impl<T> Sub for Matrix<T>
   where
      T: Clone + Sub<Output = T>,
   {
      type Output = Result<Self,&'static str>;

      fn sub(self, other: Self) -> Result<Self, &'static str> {
      if self.rows == other.rows && self.cols == other.cols {
            let data = self
               .data
               .iter()
               .zip(other.data.iter())
               .map(|(a, b)| a.clone() - b.clone())
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

/// Implement indexing of a generic matrix; a = matrix[(i,j)]
impl<T> Index<(u128,u128)> for Matrix<T>
   {
      type Output = T;

      fn index(&self, coord:(u128,u128)) -> &T {
         &self.data[((self.cols * (coord.0 - 1)) + coord.1 - 1) as usize]
      }
   }

/// Implement mutable indexing of a generic matrix; matrix[(i,j,)] = a
impl<T> IndexMut<(u128,u128)> for Matrix<T>
   {
      fn index_mut(&mut self, coord:(u128,u128)) -> &mut Self::Output {
         &mut self.data[((self.cols * (coord.0 - 1)) + coord.1 - 1) as usize]
      }
   }

/// Implement == comparison for matrices
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



/// Implement standard functions for generic matrices
impl<T: Clone + Copy> Matrix<T>
{
   /// Constructor for a new matrix from Vec
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

   /// Get nth row as Vec
   #[allow(dead_code)]
   pub fn row(&self, idx: u128) -> &[T] {
      &self.data[(self.cols*(idx-1)) as usize..((self.cols*(idx-1)) + self.cols) as usize]
   }

   /// Get nth col as Vec
   pub fn col(&self, idx: u128) -> Vec<T> {
                        self
                        .data
                        .iter()                  // Create an iterator over the vector
                        .skip((idx - 1) as usize)                 // Skip the first elements
                        .step_by(self.cols as usize)              // Take every nth element
                        .cloned()
                        .collect()
                     
   }

   /// Get diagonal as Vec
   #[allow(dead_code)]
   pub fn diag(&self) -> Vec<T> {

      assert_eq!(self.cols, self.rows, "Matrix must be square.");

      let mut counter: usize = 0;
      let mut data:Vec<T> = Vec::new();
      for (index,val) in self.data.iter().enumerate() {
         if index == (counter * (self.cols as usize) + counter) {
            data.push(*val);
            counter += 1;
         }
      }  
      data
   }

   /// The general term for any diagonal going top-left to bottom-right direction is k-diagonal 
   /// where k is an offset form the main diagonal. k=1 is the superdiagonal, k=0 is the main diagonal, and k=−1
   /// is the subdiagonal. According to Mathworld, the general term for the antidiagonals seems to be skew-diagonals.
   pub fn skew_diag(&self,offset: i32) -> Vec<T>
   where 
      T: Clone {
      assert_eq!(self.cols, self.rows, "Matrix must be square.");
      let data: Vec<T> = self.data
                              .iter()
                              .skip( if offset < 0 { offset.abs() as usize * self.cols as usize  } else { offset as usize})
                              .step_by((self.cols+1) as usize)
                              .cloned()
                              .collect(); 
      data
   }

   fn round_to_decimal_places(value: T, decimal_places: u32) -> T
   where 
      T: Float {
      let factor = T::from(10).unwrap().powi(decimal_places as i32);
      (value * factor).round() / factor
   }
   
   /// Are all the elements below the diagonal zero to a given precision.
   pub fn is_upper_triang(&self, precision: f64) -> bool
   where 
      T: Clone + Float {
      assert_eq!(self.cols, self.rows, "Matrix must be square.");
      
      self.data
            .iter()
            .enumerate()
            .skip( self.cols as usize)
            .step_by((self.cols) as usize)
            .filter_map(|(start_idx, _)| {
               if start_idx < self.data.len() - self.cols as usize {
                  Some(self.data[start_idx..start_idx + (start_idx/self.cols as usize) - 1].to_vec())
               } else {
                  None
               }})
            .flatten()
            .all(|x|   Matrix::<T>::round_to_decimal_places(x,12) <  T::from(precision).unwrap())
   }

   /// Are all the elements below the diagonal zero to a given precision.
   pub fn is_lower_triang(&self, precision: f64) -> bool
   where 
      T: Clone + Float {
      assert_eq!(self.cols, self.rows, "Matrix must be square.");
      
      self.data
            .iter()
            .enumerate()
            .step_by((self.cols) as usize)
            .filter_map(|(start_idx, _)| {
               if start_idx < (self.data.len() - self.cols() as usize) {
                  Some(self.data[(start_idx  + (start_idx/self.cols as usize + 1))..(start_idx + self.cols as usize)].to_vec())
               } else {
                  None
               }})
            .flatten()
            .all(|x| Matrix::<T>::round_to_decimal_places(x,12) <  T::from(precision).unwrap())
   }

   /// Construct a n x n identity matrix
   ///
   /// # Arguments
   ///
   /// * `dimen` - f64.
   ///
   /// # Returns
   ///
   /// The `dimen` x `dimen` identity matrix of type T.
   ///
   /// Example
   /// ```
   /// use crum::complex::Complex;
   /// use crum::matrix::Matrix;
   /// let m_id = Matrix::<Complex<f64>>::identity(10);
   /// assert!(Matrix::<Complex<f64>>::is_identity(&m_id, 1e-15));
   /// assert_eq!(m_id.rows(),m_id.cols())
   /// 
   /// ```
   pub fn identity(dimen: usize) -> Self
   where 
      T: Zero + One      
   {
      let mut identity = Matrix::new(dimen as u128,dimen as u128, vec![T::zero(); dimen*dimen]);
      let mut counter: usize = 0;
      for (index,val) in identity.data.iter_mut().enumerate() {
         if index == (counter * dimen + counter) {
            *val = T::one();
            counter += 1;
         }
      }  
      identity
   }

   /// Check if matrix is identity matrix by confirming that main diagonal is identity 1
   /// and it satisfies both upper and lower triangular requirements.
   ///
   /// # Arguments
   ///
   /// * `&self` - the matrix.
   /// * `precision` - f64.
   ///
   /// # Returns
   ///
   /// The bool result comparing with T::zero() and T::one() to required precision.
   ///
   /// # Example
   /// ```
   /// use crum::complex::Complex;
   /// use crum::matrix::Matrix;
   /// let m_identity_4_4 = Matrix::<Complex<f64>>::identity(4);
   /// assert!(m_identity_4_4.is_identity(1e-12)); 
   /// 
   /// ```
   #[allow(dead_code)]
   pub fn is_identity(&self,precision: f64) -> bool
   where 
      T: One + PartialEq + Float
   {     
      assert_eq!(self.cols, self.rows, "Matrix must be square.");
      if self.is_lower_triang(precision ) && self. is_upper_triang(precision) {
         (self.diag()).iter().all(|&x| Matrix::<T>::round_to_decimal_places( (T::one() - x).abs(),12) <  T::from(precision).unwrap() )
      } else {
         false
      }      
   }

   /// Set nth row of a matrix
   ///
   /// # Arguments
   ///
   /// * `self` - Mutable reference to this matrix.
   /// * `idx` - Row index to be replaced, 1-based.
   /// * `row` - Vector of type T to replace row in matrix.
   ///
   /// # Returns
   ///
   /// # Example
   /// ```
   /// use crum::matrix;
   /// use crum::complex::Complex;
   /// use crum::matrix::Matrix;
   /// let mut m_complex_f  = matrix![[Complex::new(0.0, 0.0), Complex::new(6.1, -4.0), Complex::new(3.0, -4.0)],
   ///                                 [Complex::new(6.1, 4.0), Complex::new(1.0, 0.0), Complex::new(2.0, -5.0)],
   ///                                 [Complex::new(3.0, 4.0), Complex::new(2.0, 5.0), Complex::new(2.0, 0.0)]];
   /// let replacement = vec![Complex::new(5.5, 6.6), Complex::new(4.0, 0.0), Complex::new(2.0, 9.0)];
   /// m_complex_f.row_set(2,replacement);
   /// assert!(m_complex_f[(2,1)] == Complex::new(5.5, 6.6) 
   ///      && m_complex_f[(2,2)] == Complex::new(4.0, 0.0) 
   ///      && m_complex_f[(2,3)] == Complex::new(2.0, 9.0));
   /// 
   /// ```
   #[allow(dead_code)]
   pub fn row_set(&mut self, idx: u128, row: Vec<T>)
   {
      assert_eq!(row.len(),self.cols() as usize, "Replacement row must be of length number of matrix columns.");
      let _ = &self.data.splice((self.cols *(idx - 1)) as usize..((self.cols *(idx - 1))+self.cols) as usize,row);     
   }   

   /// Set nth col of a matrix
   ///
   /// # Arguments
   ///
   /// * `self` - Mutable reference to this matrix.
   /// * `idx` - Column index to be replaced, 1-based.
   /// * `row` - Vector of type T to replace column in matrix.
   ///
   /// # Returns
   ///
   /// # Example
   /// ```
   /// use crum::matrix;
   /// use crum::complex::Complex;
   /// use crum::matrix::Matrix;
   /// let mut m_complex_f  = matrix![[Complex::new(0.0, 0.0), Complex::new(6.1, -4.0), Complex::new(3.0, -4.0)],
   ///                                 [Complex::new(6.1, 4.0), Complex::new(1.0, 0.0), Complex::new(2.0, -5.0)],
   ///                                 [Complex::new(3.0, 4.0), Complex::new(2.0, 5.0), Complex::new(2.0, 0.0)]];
   /// let replacement = vec![Complex::new(5.5, 6.6), Complex::new(4.0, 0.0), Complex::new(2.0, 9.0)];
   /// m_complex_f.col_set(2,replacement);
   /// assert!(m_complex_f[(1,2)] == Complex::new(5.5, 6.6) 
   ///      && m_complex_f[(2,2)] == Complex::new(4.0, 0.0) 
   ///      && m_complex_f[(3,2)] == Complex::new(2.0, 9.0));
   /// 
   /// ``` 
   #[allow(dead_code)]
   pub fn col_set(&mut self, idx: u128, col: Vec<T>)
      {    
         assert_eq!(col.len(),self.rows() as usize, "Replacement column must be of length number of matrix rows.");
         let vec_iter = self.data
         .iter_mut()  // mutable iterator over the vector
         .skip((idx - 1) as usize)  // Skip the first elements
         .step_by(self.cols as usize);
         
         let mut index:usize = 0;
         for x in vec_iter{
            *x = col[index].clone();
            index += 1;
         }
   }

   /// Get matrix data as Vec; row dominant
   #[allow(dead_code)]
   pub fn data(&self) -> Vec<T> {
      self.data.clone()
   }

   /// Get matrix data as Vec; row dominant
   #[allow(dead_code)]
   pub fn rows(&self) -> u128 {
      self.rows
   }

         /// Get matrix data as Vec; row dominant
   #[allow(dead_code)]
   pub fn cols(&self) -> u128 {
      self.cols
   }

   /// Transpose of a matrix
   ///
   /// # Arguments
   ///
   /// * `self` - This matrix.
   ///
   /// # Returns
   ///
   /// The new transposed matrix.
   ///
   /// # Example
   /// ```
   /// use crum::matrix;
   /// use crum::complex::Complex;
   /// use crum::matrix::Matrix;
   /// let mat = matrix![
   ///    [1.0, 2.0, 7.0],
   ///    [3.0, 4.0, 8.0]];
   /// let mat_trans = mat.trans();
   /// 
   /// assert!(mat_trans[(3,1)] == 7.0
   ///      && mat_trans[(3,2)] == 8.0);
   /// 
   /// ``` 
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

   /// Extract sub-matrix from matrix by specifying a row range and column range
   /// ///
   /// # Arguments
   ///
   /// * `self` - This matrix.
   /// * `rows` - Inclusive range of one-based row indices.
   /// * `cols` - Inclusive range of one-based column indices.
   ///
   /// # Returns
   ///
   /// The new sub-matrix.
   ///
   /// # Example
   /// ```
   /// use crum::matrix;
   /// use crum::complex::Complex;
   /// use crum::matrix::Matrix;
   /// let m_f64 = matrix![[      0.4130613594781728,      0.06789616787771548,     0.9656690602669977,      0.935185936307158,       0.36917500405507325],  
   ///                     [       0.243168572454519     ,  0.31261293138410245   ,  0.5252879127056232   ,   0.0330935153674985    ,  0.9753704689278127],  
   ///                     [       0.41435210063956374   ,  0.2399665922965562    ,  0.9561861714688091   ,   0.697771062293815     ,  0.010276832821779937],
   ///                     [       0.011234519442551607  ,  0.3769729167821171    ,  0.3831601028613202   ,   0.9278273814572322    ,  0.5884363505264488],  
   ///                     [       0.22699699307514926   ,  0.30855453813133443   ,  0.704016327682634    ,   0.9993467239797109    ,  0.5789833380097665],  
   ///                     [       0.46588910028585306   ,  0.5638014697165844    ,  0.17953318066016835  ,   0.8848724908721202    ,  0.31679387457452873], 
   ///                     [       0.6519277647881355    ,  0.38278755861532476   ,  0.537292424163817    ,   0.6468661089689082    ,  0.34068558363646023], 
   ///                     [       0.554723676204958     ,  0.7331917287295512    ,  0.4119117955980295   ,   0.03440648890443177   ,  0.2875483259617025],  
   ///                     [       0.35148199904388605   ,  0.06203799828471969   ,  0.3318123341812714   ,   0.474823312748173     ,  0.6748846353426651],  
   ///                     [       0.35208686528649485   ,  0.08035514071119178   ,  0.9146101290660322   ,   0.20577219168723354   ,  0.6125322396803093]]; 
   /// let m_f64_sub = m_f64.sub_matrix(3..=7, 2..=4);
   /// assert!(m_f64_sub.rows() == 5 && m_f64_sub.cols() == 3);
   /// assert!(m_f64_sub[(1,1)] == 0.2399665922965562 && m_f64_sub[(5,3)] == 0.6468661089689082);
   /// ```  
   pub fn sub_matrix(&self, rows: RangeInclusive<u128>,cols: RangeInclusive<u128> ) -> Self
   where 
      T: Zero
      {
         assert!(1 <= *rows.start(),"Row range start >= 1.");
         assert!(self.rows >= *rows.end(),"Row range end {} <= Number of Rows {}.",*rows.end(), self.rows);
         assert!(1 <= *cols.start(),"Col range start >= 1.");
         assert!(self.cols >= *cols.end(),"Col range end <= Number of Columns.");

         let skip = (self.cols * (rows.start()-1) + (cols.start() - 1)) as usize;
         let extracted_data:Vec<T> = self.data
         .iter()
         .enumerate()
         .skip(skip)
         .step_by(self.cols as usize)
         .filter_map(|(start_idx, _)| {
            if start_idx < ((rows.clone().count() * self.cols as usize)+skip) {
               Some(self.data[start_idx..start_idx + cols.clone().count()].to_vec())
            } else {
               None
            }
         })
         .flatten()
         .collect();
      //println!("rows {:?} \ncols {:?} \ninput {} \ndata {:?}", rows,cols,self,extracted_data);
      Self::new(rows.count() as u128, cols.count() as u128, extracted_data)
   }

   /// Extend a matrix by inserting a row immediately after idx
   /// If idx == \<u128\>::max_value() then insert as index 1
   pub fn insert_row(self, idx: u128, row: Vec<T>) -> Self
   {
      //assert!(1 <= idx && idx <= self.rows,"Row insert 1 <= {} <= {} in matrix.",idx,self.rows);
      assert_eq!( row.len(), self.cols as usize,"Row vector must have same dimension and matrix columns.");
      
      let index = if idx == <u128>::max_value() {0} else {(self.cols * idx) as usize};
      let mut data = self.data;
      data.splice(index..index,row.clone());
      Self { rows: self.rows + 1, cols: self.cols, data: data }
   }

   /// Extend a matrix by inserting a col immediately after idx
   /// If idx == \<u128\>::max_value() then insert as index 1
   pub fn insert_col(self, idx: u128, col: Vec<T>) -> Self
   {
      //assert!(1 <= idx && idx <= self.cols,"Col insert 1 <= index <= number of cols in matrix.");
      assert_eq!( col.len(), self.rows as usize,"Column vector must have same dimension and matrix rows.");

      let mut index:usize = if idx == <u128>::max_value() {0} else {idx as usize};
      let mut data = self.data;
      for x in col{
         data.insert(index,x.clone());
         index += (self.cols + 1) as usize;
      }
      Self { rows: self.rows, cols: self.cols + 1, data: data }
   }

   /// Block Matrix Augmentation: Adding an identity sub-matrix to the top-left corner of a larger matrix, 
   /// while padding the rest with zeros, creates a block matrix. This process could also be considered 
   /// a form of direct sum in certain contexts.
   pub fn augment(self, id_dimen: u128) -> Self
   where 
      T: Zero + One + PartialEq
      {
         let mut aug_mat = self;
         for _i in 1..=id_dimen {
            aug_mat = aug_mat.clone()
               .insert_row(<u128>::max_value(), vec![<T>::zero(); aug_mat.cols as usize])
               .insert_col(<u128>::max_value(), nth_identity_vector(1, aug_mat.rows as usize + 1) );           
         }
         aug_mat
   }

   /// Determinant of a 2x2 Matrix
   pub fn det_2x2(self) -> T
   where 
      T: Mul<Output = T> + Sub<Output = T> {
      (self.data[0]*self.data[3]) - (self.data[1]*self.data[2])
   }

   /// Convert Real to Complex Matrix
   pub fn to_complex(&self) -> Matrix<Complex<T>>
   where 
      T: Zero + Float
   {
      let data: Vec<Complex<T>> = self.data
                  .iter()
                  .map(|x| Complex::<T>::new(*x, T::zero()))
                  .collect();

      Matrix::<Complex<T>>::new(self.rows,self.cols,data)
   }

   #[allow(dead_code)]
   pub fn rnd_matrix( rows: u128, cols: u128, rnd: RangeInclusive<T> ) -> Self
   where 
      T: SampleUniform + PartialOrd {
      let mut rng = rand::thread_rng(); // Create a thread-local RNG
      let size = rows * cols;

      // Fill the vector with random numbers in defined range
      let random_numbers = (0..size)
         .map(|_| rng.gen_range(rnd.clone()) )
         .collect();

      Matrix::new(rows,cols,random_numbers)
   }

   pub fn swap_rows(&mut self, row1_idx: usize, row2_idx:usize)
   where 
      T: Float
   {
      assert!( (((row1_idx-1)*self.cols as usize)+ self.cols as usize) <= self.data.len(),"Matrix dimensions exceeded! {} <= {}",(((row1_idx-1)*self.cols as usize)+ self.cols as usize),self.data.len());
      assert!( (((row2_idx-1)*self.cols as usize)+ self.cols as usize) <= self.data.len(),"Matrix dimensions exceeded! {} <= {}",(((row2_idx-1)*self.cols as usize)+ self.cols as usize),self.data.len());

      // row2 -> row1
      let row1:Vec<T> = self.data.splice( (row1_idx-1)*self.cols as usize..((row1_idx-1)*self.cols as usize) + self.cols as usize,
      self.data[((row2_idx-1) * self.cols as usize)..((row2_idx-1) * self.cols as usize) + self.cols as usize].to_vec()).collect();
      // row1 -> row2
      self.data.splice(((row2_idx-1)*self.cols as usize)..((row2_idx-1)*self.cols as usize) + self.cols as usize,
      row1);      
   }


   /* INPUT: A,P filled in LUPDecompose; b - rhs vector; N - dimension
 * OUTPUT: x - solution vector of A*x=b
 */
// void LUPSolve(double **A, int *P, double *b, int N, double *x) {

//    for (int i = 0; i < N; i++) {
//        x[i] = b[P[i]];

//        for (int k = 0; k < i; k++)
//            x[i] -= A[i][k] * x[k];
//    }

//    for (int i = N - 1; i >= 0; i--) {
//        for (int k = i + 1; k < N; k++)
//            x[i] -= A[i][k] * x[k];

//        x[i] /= A[i][i];
//    }
// }
   pub fn linear_solve_lu(l: &Matrix<T>, u: Matrix<T>, p: Matrix<T>, b: &Vec<T>) -> Vec<T>
   where 
      T: Float
   {
      // Lx = Py
      let mut x:Vec<T> = p.data.chunks(p.cols as usize)
                     .map(|x | b[x.iter().position(|i| i.is_one()).unwrap()])
                     .collect();
      
      x.iter_mut().enumerate().map(|a| *a = *a -    );



      x
   }

   /// Determinant of a real square matrix using its U from 
   /// A = LU decomposition(Gauss Elimination with Partial Pivot - GEPP)
   /// 
   /// # Arguments
   ///
   /// * `self` - Upper Triangular component of a LU decomposition using GEPP.
   /// * `swaps` - Number of row swaps during GEPP.
   ///
   /// # Returns
   ///
   /// The determinant of the original matrix A.
   ///
   /// #Example   
   /// ```
   /// use crum::matrix::Matrix;
   /// use crum::matrix;
   /// let mut m_a = matrix![[0.0,5.0,22.0/3.0],
   ///                       [4.0,2.0,1.0],
   ///                       [2.0,7.0,9.0]];
   /// let (l,u,p, swaps) = m_a.lu(1e-15).unwrap();
   /// assert_eq!(num_traits::Float::round(u.determinant_lu(swaps)), 6.0);
   /// ```
   pub fn determinant_lu(&self, swaps: u32) -> T
   where 
      T: Float
      {
         let det = self.diag().iter().fold(T::one(),|acc,x| acc * *x );
         if swaps % 2 == 0 {det} else {-det} 
      }
   
   /// LU Decompose a real matrix using Gauss Elimination with Partial Pivot.
   /// Output is L,U,P such that P*A=L*U.
   /// 
   /// # Arguments
   ///
   /// * `self` - Real square matrix.
   /// * `precision` - Tolerance to determine if matrix is degenerate.
   ///
   /// # Returns
   ///
   /// Matrices L(Lower Triangular),U(Upper Triangular),P(Permutation) and the number of rows swaps in the pivot process.
   ///
   /// #Example   
   /// ```
   /// use crum::matrix::Matrix;
   /// use crum::matrix;
   /// let mut m_a = matrix![[0.0,5.0,22.0/3.0],
   ///                       [4.0,2.0,1.0],
   ///                       [2.0,7.0,9.0]];
   /// let (l,u,p, swaps) = m_a.lu(1e-15).unwrap();
   /// let lhs = p.clone() * m_a.clone();
   /// let rhs = l* u;
   /// assert!(lhs.data().iter().zip(rhs.data().iter()).all(|(x,y)| x == y ));
   /// ```
   pub fn lu(&self, precision: f64) -> Result<(Self,Self,Self,u32),String> 
   where
      T: Float + Display + Debug,
   {
      let mut mat_a = self.clone();
      let mut mat_p = Matrix::<T>::identity(self.rows as usize); // permutation matrix
      let mut rows_swaps = 0_u32;
      let ncols = self.cols as usize;
      let nrows = self.rows as usize;
      let mut mat_l = Matrix::<T>::new(nrows as u128,ncols as u128,vec![T::zero();ncols*nrows]); // lower triangular matrix

      assert_eq!(ncols,nrows,"Matrix must be square.");

      // Start Iteration
      for diag_idx in 1..ncols {

         // check for max value in relevant column
         let (max_row_idx_vec, max_diag_elem) = mat_a.data.iter()
                                                                     .enumerate()
                                                                     .skip(diag_idx-1 + (diag_idx-1)*ncols)
                                                                     .step_by(ncols)
                                                                     .reduce(|acc,x| if acc.1.abs() < x.1.abs() {x} else {acc})
                                                                     .unwrap();
         let max_row_idx = (max_row_idx_vec/nrows) + 1;
         let max_element = *max_diag_elem;
         
         
         if max_element.to_f64().unwrap() > precision {
            if diag_idx != max_row_idx {
               // Swap k row with max row so that a_diag_idxdiag_idx is max and non zero
               let top_row:Vec<T> = mat_a.data.splice(((max_row_idx-1) * ncols)..((max_row_idx-1) * ncols) + ncols,
               mat_a.data[(diag_idx-1)*ncols..((diag_idx-1)*ncols) + ncols].to_vec()).collect();               
               mat_a.data.splice((diag_idx-1)*ncols..((diag_idx-1)*ncols) + ncols,top_row.clone());      
               
               // Swap same rows in identity matrix p
               let row1:Vec<T> = mat_p.data.splice( (diag_idx-1)*ncols..((diag_idx-1)*ncols) + ncols,
               mat_p.data[((max_row_idx-1) * ncols)..((max_row_idx-1) * ncols) + ncols].to_vec()).collect();               
               mat_p.data.splice(((max_row_idx-1)*ncols)..((max_row_idx-1)*ncols) + ncols, row1);      

               rows_swaps += 1;

               // Swap same rows in L matrix
               let row1:Vec<T> = mat_l.data.splice( (diag_idx-1)*ncols..((diag_idx-1)*ncols) + ncols,
               mat_l.data[((max_row_idx-1) * ncols)..((max_row_idx-1) * ncols) + ncols].to_vec()).collect();               
               mat_l.data.splice(((max_row_idx-1)*ncols)..((max_row_idx-1)*ncols) + ncols, row1);      


               // adjust ALL rows in BENEATH a_diag_idxdiag_idx as follows:
               // row_k = row_k - row_1 * a_k1/a_11            
               for beneath_diag_idx in diag_idx..ncols {
                  mat_l[((beneath_diag_idx+1) as u128,(diag_idx) as u128)] = mat_a[((beneath_diag_idx+1) as u128,(diag_idx) as u128)]/max_element;
                  let mut replacement_row_iter = top_row.iter()
                                                                     .enumerate()                                                                  
                                                                     .zip(mat_a.data.iter_mut().skip(((beneath_diag_idx-1)*ncols)+ ncols));


                  while let Some(elem) = replacement_row_iter.next()  {
                     *elem.1 = *elem.1 - (mat_l[((beneath_diag_idx+1) as u128,(diag_idx) as u128)] * *elem.0.1);
                  }
               }
            } else {
               // Proceed without row swap
               let top_row:Vec<T> =  mat_a.row(diag_idx as u128).iter().cloned().collect();
               // adjust ALL rows in BENEATH a_diag_idxdiag_idx as follows:
               // row_k = row_k - row_1 * a_k1/a_11            
               for beneath_diag_idx in diag_idx..ncols {
                  mat_l[((beneath_diag_idx+1) as u128,(diag_idx) as u128)] = self[((beneath_diag_idx+1) as u128,(diag_idx) as u128)]/max_element;
                  let mut replacement_row_iter = top_row.iter()
                                                                     .enumerate()                                                                  
                                                                     .zip(mat_a.data.iter_mut().skip(((beneath_diag_idx-1)*ncols)+ ncols));


                  while let Some(elem) = replacement_row_iter.next()  {
                     *elem.1 = *elem.1 - (mat_l[((beneath_diag_idx+1) as u128,(diag_idx) as u128)] * *elem.0.1);
                  }
               }
            }
         } else {
            return Err("Matrix is degenerate.".to_string());
         }
      }

      // update diagonal of mat_l with ones
      let _ = mat_l.data.iter_mut().step_by(ncols+1).for_each(|x| *x = T::one());
 
      Ok((mat_l,mat_a,mat_p,rows_swaps))
   }

   // pub fn eigen_schur(&self) -> (Vec<T>,Vec<Complex<T>>)
   // where 
   //    T: Float + From<f64>,
   //    f64: From<T> + Mul<T>
   // {
   //    assert_eq!(self.rows,self.cols,"Matrix must be a square matrix");
   //    let eigen_values_real = Vec::<T>::new();
   //    let eigen_values_complex = Vec::<Complex<T>>::new();


   //    // // set real-complex split threshold
   //    // let threshold = Complex::new(<T as From<f64>>::from(0.001),<T as From<f64>>::from(0.001)).magnitude();

   //    // // Process diagonal : if block then complex conjugate eigen pair, if not then real eigen value
   //    // // For each elem(ij), if elem(i+1,j) == zero, then elem(ij) is a real eigen value, and skip 1 diag elem, else;
   //    // // the 2x2 sub-matrix i..i+1 x j..j+1 has complex conjugate eigenvalue pairs.

   //    // let mut row_idx = 1_u128;
   //    // let mut col_idx = 1_u128;

   //    // while row_idx <= self.rows {
   //    //    while col_idx <= self.cols {

   //    //       if row_idx == self.rows && col_idx == self.cols {
   //    //          //println!("eigen real: {}",schur[(row_idx,col_idx)]);
   //    //          eigen_values_real.push(self[(row_idx,col_idx)].real());
   //    //          row_idx += 1;
   //    //          col_idx += 1;
   //    //       } else{
   //    //          if self[(row_idx+1,col_idx)].magnitude() <= threshold {
   //    //             //println!("eigen real: {}",schur[(row_idx,col_idx)]);
   //    //             eigen_values_real.push(self[(row_idx,col_idx)].real());
   //    //             row_idx += 1;
   //    //             col_idx += 1;

   //    //          } else {
   //    //             let schur_sub = self.sub_matrix(row_idx as u128..= (row_idx+1)  as u128, col_idx as u128..=(col_idx+1) as u128);
   //    //             let (lambda1,lambda2) = Matrix::eigen_2x2(schur_sub);
   //    //             // Check if we have a complex conjugate pair
   //    //             if lambda1.conj() == lambda2 {
   //    //                //println!("eigen complex: {} {}",lambda1,lambda2 );
   //    //                eigen_values_complex.push(lambda1);
   //    //                eigen_values_complex.push(lambda2);
   //    //                row_idx += 2; col_idx += 2;
   //    //             } else {
   //    //                eigen_values_real.push(self[(row_idx,col_idx)].real());
   //    //                row_idx += 1;
   //    //                col_idx += 1;                     
   //    //             }
   //    //          }
   //    //       }
   //    //    }
   //    // }
   //    (eigen_values_real,eigen_values_complex)
   // }

   #[allow(dead_code)]
   fn mul(self, other: T) -> Self
   where 
      T: Float {
      Self {
         rows: self.rows,
         cols: self.cols,
         data: self.data.iter().map(|&x| x*other).collect()
      }
   }

}



/// Complex Matrix Specializations
impl<T> Matrix<Complex<T>>
   where
      Matrix<Complex<T>>: PartialEq,
      T: Clone + Float + std::ops::Neg<Output = T>
{
   /// norm of x of degree two(Frobenius Norm) = ||x||2 = ( x1 x∗1 + . . . + xk x∗k ).sqrt on complex numbers only
   ///    
   /// ```
   /// use crum::matrix::Matrix;
   /// use crum::complex::Complex;
   /// let result2 = Matrix::<Complex<f64>>::norm_2(&vec![Complex::new(5.0, 3.0),Complex::new(2.0, 4.0),Complex::new(7.0, 1.0),Complex::new(9.0, 5.0)]);
   /// assert_eq!(result2, Complex::new(14.491376746189438, 0.0));
   /// ```
   pub fn norm_2(vec: &Vec<Complex<T>>) -> Complex<T>
   where
      T: Float + From<f64>,
      f64: From<T>
   {
      let sum = vec.iter().fold(Complex::<T>::zero(), |acc, x| acc + *x * x.conj());
      sum.sqrt()
   }

   /// norm of x of degree two(Frobenius Norm) of a complex matrix
   ///    
   /// ```
   /// use crum::matrix::Matrix;
   /// use crum::complex::Complex;
   /// ```
   pub fn norm_2_mat(&self) -> Complex<T>
   where
      T: Float + From<f64>,
      f64: From<T>
   {
      let sum = self.data.iter().fold(Complex::<T>::zero(), |acc, x| acc + acc + *x * x.conj());
      sum.sqrt()
   }

   /// ```
   /// use crum::matrix::Matrix;
   /// use crum::complex::Complex;
   /// let result2 = Matrix::<Complex<f64>>::dot_product(&vec![Complex::new(5.0, 3.0),Complex::new(2.0, 4.0)],&vec![Complex::new(7.0, 1.0),Complex::new(9.0, 5.0)]);
   /// assert_eq!(result2, Complex::new(76.0, -42.0));
   /// ```
   pub fn dot_product(v1: &Vec<Complex<T>>, v2: &Vec<Complex<T>>) -> Complex<T>
   where
      T: Copy + Zero + Mul<Output = T>
   {
      v1.iter().zip(v2).fold(Complex::<T>::zero(), |acc, (&x, &y)| acc + x.conj() * y)
   }
   

   /// Generate a complex matrix of a given dimension with randomized variable over a uniform distribution
   /// 
   /// # Arguments
   ///
   /// * `rows` - Number of rows as u128.
   /// * `cols` - Number of cols as u128.
   /// * `rnd`  - Random number inclusive range.
   ///
   /// # Returns
   ///
   /// The new matrix.
   ///
   /// #Example   
   /// ```
   /// use crum::matrix::Matrix;
   /// use crum::complex::Complex;
   /// let m_f64 = Matrix::<Complex<f64>>::rnd_complex_matrix(10, 9, 0.0..=1.0);
   /// assert_eq!(m_f64.rows(), 10);
   /// assert_eq!(m_f64.cols(), 9);
   /// assert_eq!(m_f64.data().len(), 90);
   /// ```
   pub fn rnd_complex_matrix( rows: u128, cols: u128, rnd: RangeInclusive<T> ) -> Self
   where 
      T: SampleUniform {
      let mut rng = rand::thread_rng(); // Create a thread-local RNG
      let size = rows * cols; // Define the size of the vector

      // Fill the vector with random numbers between 0 and 100
      let random_numbers: Vec<Complex<T>> = (0..size)
         .map(|_| Complex::new(rng.gen_range(rnd.clone()),rng.gen_range(rnd.clone())) ) // Generate numbers in the range [0, 100]
         .collect();

      Matrix::new(rows,cols,random_numbers)
   }

   /// Get the complex conjugate of a complex vector
   /// 
   /// # Arguments
   ///
   /// * `v` - Complex vector Vec of type Complex\<T\>.
   ///
   /// # Returns
   ///
   /// Complex vector Vec of type Complex\<T\>.
   ///
   /// #Example   
   /// ```
   /// use crum::matrix::Matrix;
   /// use crum::complex::Complex;
   /// let v_f64 = Matrix::<Complex<f64>>::vec_conj(vec![Complex::new(5.0, 3.0),Complex::new(2.0, -4.0),Complex::new(7.0, 1.0),Complex::new(9.0, -5.0)]);
   /// assert_eq!(v_f64, vec![Complex::new(5.0, -3.0),Complex::new(2.0, 4.0),Complex::new(7.0, -1.0),Complex::new(9.0, 5.0)]);
   /// ```
   pub fn vec_conj(v: Vec<Complex<T>>) -> Vec<Complex<T>> {
      v.iter().map(|x| Complex::new(x.real(),-x.imag())).collect()
   }
   
   /// Get the complex conjugate of a complex matrix
   /// 
   /// # Arguments
   ///
   /// * `self` - This complex matrix.
   ///
   /// # Returns
   ///
   /// New complex matrix of type Complex\<T\>.
   ///
   /// #Example   
   /// ```
   /// use crum::matrix::Matrix;
   /// use crum::complex::Complex;
   /// ```
   pub fn conj(self) -> Self {
      Self {
         rows: self.cols,
         cols: self.rows,
         data: self.data.iter().map(|x| Complex::new(x.real(),-x.imag())).collect()
      }
   }

   /// Check if a complex matrix is Hermitian/Self-adjoint - Charles Hermite 1855
   #[allow(dead_code)]
   pub fn is_hermitian(self) -> bool {
      if self.clone().trans().conj() == self {
         true
      } else {
         false
      }
   }

   /// Do complex QR-decomposition using complex Householder Transforms
   /// Require rows >= cols.
   /// 
   /// # Arguments
   ///
   /// * `self` - This complex matrix.
   ///
   /// # Returns
   ///
   /// New complex matrix pair Q and R, where Q is unitary and R is upper trangular.
   ///
   /// #Example
   /// ```
   /// use crum::matrix;
   /// use crum::matrix::Matrix;
   /// use crum::complex::Complex;
   /// let m_complex_f64 = matrix![[      Complex::new(0.5833556123799982,0.5690181027600784),    Complex::new(0.6886043600138049,0.674390821408502),     Complex::new(0.24687850063786915,0.5935898903765723),   Complex::new(0.00933456816360523,0.6587484783595824),   Complex::new(0.23512331858462204,0.15986594969605908),      Complex::new(0.3592667599232367,0.044091292164025304),  Complex::new(0.9128331393696729,0.1833852110584138),    Complex::new(0.5180466720472582,0.05333044453605408),   Complex::new(0.26564558002149125,0.24744281386070038),  Complex::new(0.5795439760266531,0.7097323035461603)],
   ///                             [       Complex::new(0.19355221625091562,0.07443815946182132),  Complex::new(0.38523666576656257,0.6235838654566793),   Complex::new(0.5655998866671316,0.02796381067764698),   Complex::new(0.9478369597737368,0.5061665241108549),    Complex::new(0.9665277542211836,0.6464090293919905),Complex::new(0.8934413145999256,0.9928347855455917),    Complex::new(0.24012630465410162,0.4511339192624414),   Complex::new(0.0795066100131381,0.16804618159775966),   Complex::new(0.7154655006062255,0.27954112740219067),   Complex::new(0.8093795163995636,0.2647562871405445)     ],
   ///                             [       Complex::new(0.722107408913046,0.3453081577838271),     Complex::new(0.8641742059855633,0.43503554725558835),   Complex::new(0.6576465324620399,0.07852371724975284),   Complex::new(0.38540857835795383,0.5959496185548973),   Complex::new(0.617794405196579,0.7206737924044645),Complex::new(0.5099081560147524,0.8617303081795931),     Complex::new(0.4464823442359144,0.45949602324474303),   Complex::new(0.35752646144365713,0.8983136848274984),   Complex::new(0.6077708116013137,0.6456302985283519),    Complex::new(0.15132177386470594,0.3335043031018719)    ],
   ///                             [       Complex::new(0.8593850791476851,0.11354522137267689),   Complex::new(0.4938670101809314,0.7852216948857795),    Complex::new(0.23096625492908301,0.20830882216670715),  Complex::new(0.7400234625821275,0.4639399244725205),    Complex::new(0.5378937891759045,0.7037567515968549),Complex::new(0.2219390019576138,0.23209326206010134),   Complex::new(0.4558367952542631,0.9965121614070889),    Complex::new(0.39631631122312905,0.08633527991619407),  Complex::new(0.9443163419994178,0.42288025964849474),   Complex::new(0.2332370920018188,0.9922786240129193)     ],
   ///                             [       Complex::new(0.41844631685477907,0.35277367066150905),  Complex::new(0.3475123609164877,0.7101826289551204),    Complex::new(0.5730063661284887,0.48196832299859105),   Complex::new(0.6143248737217993,0.18023274893317168),   Complex::new(0.26770558087500756,0.34859620475408054),      Complex::new(0.6573603929892361,0.09347808620771915),   Complex::new(0.559359670948789,0.7870570717326266),     Complex::new(0.5996313950882705,0.0578118744896874),    Complex::new(0.32955838142838695,0.6696041764734405),   Complex::new(0.21936429159910614,0.4952850734335351)],
   ///                             [       Complex::new(0.009248256198125528,0.4527930011455513),  Complex::new(0.6849305389546144,0.12102884187585097),   Complex::new(0.4165174797405674,0.9462171528526406),    Complex::new(0.8619475196821003,0.7381751595031852),    Complex::new(0.7596450684070042,0.33049863177437794),       Complex::new(0.37937891021432113,0.0938634467267199),   Complex::new(0.046313554311124834,0.8748186202857586),  Complex::new(0.9142747660514274,0.1720666151092736),    Complex::new(0.1155038542568945,0.8407799452002931),    Complex::new(0.6036564509415651,0.1549954601684234)],
   ///                             [       Complex::new(0.5280435825255948,0.722128687058149),     Complex::new(0.2958172383395739,0.21101513922732212),   Complex::new(0.6247036944583212,0.2958591539645528),    Complex::new(0.8771245810505679,0.9876277069779977),    Complex::new(0.13778026369279453,0.2925863129303102),       Complex::new(0.6689687280137598,0.24661000255739168),   Complex::new(0.19090555403974846,0.1626089128725561),   Complex::new(0.5354601189785831,0.701258765248206),     Complex::new(0.09332523997739807,0.8382042501840465),   Complex::new(0.12250427701915115,0.7540059801921637)],
   ///                             [       Complex::new(0.641212141487291,0.12028843766751154),    Complex::new(0.5573689686134607,0.9281730407924285),    Complex::new(0.18228923797612165,0.937966839425133),    Complex::new(0.290229469308307,0.3447329019990513),     Complex::new(0.29421063475530934,0.7783426202185895),       Complex::new(0.2986058932928953,0.8937034310271619),    Complex::new(0.25710953476086523,0.896262417185731),    Complex::new(0.8847389180993006,0.254408582006775),     Complex::new(0.027617968349187068,0.29285850298797206), Complex::new(0.029426361218854565,0.6879945477458963)       ],
   ///                             [       Complex::new(0.23945753093683192,0.30616733782992506),  Complex::new(0.9881588754627659,0.6754586092988201),    Complex::new(0.6279846571645774,0.07795290055819983),   Complex::new(0.9880650206914865,0.43754117662346503),   Complex::new(0.5668252086231756,0.5654418870268184),Complex::new(0.9563427957776676,0.44960614238550123),   Complex::new(0.8250656417870632,0.5513468135978378),    Complex::new(0.9851555697862651,0.3608225406879005),    Complex::new(0.07324290749628194,0.358150639141774),    Complex::new(0.27138526608582186,0.19393235694918426)   ],
   ///                             [       Complex::new(0.08899561873929575,0.6885237168549837),   Complex::new(0.6759656028808306,0.16861078919167818),   Complex::new(0.14192833304987176,0.14780523687381544),  Complex::new(0.7793249757513581,0.7530672142302),       Complex::new(0.7685456523065339,0.3380067105229064),Complex::new(0.07062494877030058,0.406943551307159),    Complex::new(0.6391551970178856,0.5412405648127648),    Complex::new(0.5005075046812791,0.3967310304417494),    Complex::new(0.15753551096333432,0.4919072076795531),   Complex::new(0.09696298494836111,0.6188849576238623)    ]];
   /// 
   /// let (q,r) = m_complex_f64.qr_cht();
   /// assert!(r.is_upper_triang(1e-15));
   /// assert!(Matrix::<Complex<f64>>::is_identity(&(q.clone() * q.clone().conj().trans()), 1e-15));
   /// assert!(Matrix::<Complex<f64>>::is_identity(&(q.clone().conj().trans() * q.clone()), 1e-15));
   /// 
   /// ``` 
   pub fn qr_cht(&self) -> (Self,Self)
   where
   T:Copy + Zero + Float + From<f64>,
   f64: From<T> + Mul<T>
   {
      assert!(self.rows >= self.cols, "CHT for QR only valid when rows >= cols.");
      // Column by column build the Q and R matrices of A such
      // that Q R = A and Q has orthonormal column vectors 
      // and R is an upper diagonal matrix

      // For a given matrix A in the iteration;
      // Calc CHT for the first col vector
      let mut mat_a = self.clone();
      let mut mat_cht = Matrix::<Complex<T>>::householder_transform(mat_a.col(1));
      let mut mat_r = mat_cht.clone(); // first time does not need a resize.
      let mut mat_q = mat_cht.clone().trans().conj(); // complex conj transpose?

      let mut start_index = 1;      
      let cycles = (self.rows-1).min(self.cols);

      while start_index < cycles { // Remember, we have already done one iteration above
         // Begin iteration /////////////////////////////////////////////////////////////

         // Operate the CHT on the original matrix and get the sub-matrix
         mat_a = (mat_cht.clone() * mat_a.clone()).sub_matrix(start_index+1..=mat_a.rows, start_index+1..=mat_a.cols);

         // Calc CHT for first column vector and augment back to original matrix dimensions
         mat_cht = Matrix::<Complex<T>>::householder_transform(mat_a.col(1)).augment(start_index);

         // Augment A back to original dimensions
         mat_a = mat_a.augment(start_index);

         mat_q = mat_q * mat_cht.clone().trans().conj();
         mat_r = mat_cht.clone() * mat_r;

         start_index += 1;

         // End iteration ///////////////////////////////////////////////////////////////
      }
      (mat_q,mat_r*self.clone())
   }

   /// Householder Transform for Complex Matrices (CHT)
   /// Non-unique
   /// #Example
   /// ```
   /// use crum::complex::Complex;
   /// use crum::matrix::Matrix;
   /// let m_complex_f64 = vec![Complex::new(0.5833556123799982,0.5690181027600784),    Complex::new(0.6886043600138049,0.674390821408502),     Complex::new(0.24687850063786915,0.5935898903765723),   Complex::new(0.00933456816360523,0.6587484783595824),   Complex::new(0.23512331858462204,0.15986594969605908),      Complex::new(0.3592667599232367,0.044091292164025304),  Complex::new(0.9128331393696729,0.1833852110584138),    Complex::new(0.5180466720472582,0.05333044453605408),   Complex::new(0.26564558002149125,0.24744281386070038),  Complex::new(0.5795439760266531,0.7097323035461603)];
   /// let cht = Matrix::<Complex<f64>>::householder_transform(m_complex_f64);
   /// assert!(cht.clone().is_hermitian());
   /// assert!((cht.clone() * cht.clone().conj().trans()).is_identity(1e-15));
   /// 
   /// ```
   pub fn householder_transform(x: Vec<Complex<T>>) -> Self
      where
         T:Copy + Zero + Float + From<f64>,
         f64: From<T> + Mul<T>
         {            
            /* When the elements of the matrix are complex numbers, 
            it is denoted the Complex Householder Transform (CHT).
            The CHT is applied to a column vector x to zero out all
            the elements except the first one. */
            
            let x_norm_2 = Matrix::<Complex<T>>::norm_2(&x);
            let exp_jtheta_x1 = x[0]/(x[0]*x[0].conj()).sqrt();
            let mut u = x.clone();
            u[0] = x[0] + (exp_jtheta_x1 * x_norm_2);            

            let uh = Matrix::<Complex<T>>::vec_conj(u.clone());
            let uh_u: Complex<T> = dot_product(&uh, &u);

            //let uh_u_f64 = f64::from(uh_u.real());

            let u_uh: Matrix<Complex<T>> = cvec_rvec(&u,&uh);
            let u_uh_multipled_data:Vec<Complex<T>> = u_uh.data.iter().map(|x| Complex::<T>::new((Into::<T>::into(2.0)/uh_u.real()) * x.real(), (Into::<T>::into(2.0)/uh_u.real()) * x.imag())).collect();
            let mat_u_uh = Matrix::new(x.len()as u128,x.len() as u128, u_uh_multipled_data);
            let mat_cht = Matrix::<Complex<T>>::identity(x.len()) - mat_u_uh;      

            mat_cht.unwrap()
   }
   
   /// Perform Schur decomposition on a square complex matrix
   /// The complex eigenvalues of the complex matrix are the diagonal elements
   /// of the Schur transform matrix and can be retrieved by calling .diag().
   /// The Schur transform for a complex matrix is an upper triangular matrix.
   ///
   /// # Arguments
   ///
   /// * `self` - This complex matrix.
   /// * `precision` - Required trace elements precision. 
   ///
   /// # Returns
   ///
   /// A Result with Ok - complex Schur transform of the matrix, with diagonal reduced to given precision,
   /// and the associated unitary matrix with eigenvectors as columns.
   /// And Err message if maximum iterations are exceeded before convergence to required precision has occurred.
   ///
   /// #Example
   /// ```
   /// use crum::matrix;
   /// use crum::complex::Complex;
   /// use crum::matrix::Matrix;
   /// let m_complex_f64 = matrix![[      Complex::new(0.5833556123799982,0.5690181027600784),    Complex::new(0.6886043600138049,0.674390821408502),     Complex::new(0.24687850063786915,0.5935898903765723),   Complex::new(0.00933456816360523,0.6587484783595824),   Complex::new(0.23512331858462204,0.15986594969605908),      Complex::new(0.3592667599232367,0.044091292164025304),  Complex::new(0.9128331393696729,0.1833852110584138),    Complex::new(0.5180466720472582,0.05333044453605408),   Complex::new(0.26564558002149125,0.24744281386070038),  Complex::new(0.5795439760266531,0.7097323035461603)],
   ///[       Complex::new(0.19355221625091562,0.07443815946182132),  Complex::new(0.38523666576656257,0.6235838654566793),   Complex::new(0.5655998866671316,0.02796381067764698),   Complex::new(0.9478369597737368,0.5061665241108549),    Complex::new(0.9665277542211836,0.6464090293919905),Complex::new(0.8934413145999256,0.9928347855455917),    Complex::new(0.24012630465410162,0.4511339192624414),   Complex::new(0.0795066100131381,0.16804618159775966),   Complex::new(0.7154655006062255,0.27954112740219067),   Complex::new(0.8093795163995636,0.2647562871405445)     ],
   ///[       Complex::new(0.722107408913046,0.3453081577838271),     Complex::new(0.8641742059855633,0.43503554725558835),   Complex::new(0.6576465324620399,0.07852371724975284),   Complex::new(0.38540857835795383,0.5959496185548973),   Complex::new(0.617794405196579,0.7206737924044645),Complex::new(0.5099081560147524,0.8617303081795931),     Complex::new(0.4464823442359144,0.45949602324474303),   Complex::new(0.35752646144365713,0.8983136848274984),   Complex::new(0.6077708116013137,0.6456302985283519),    Complex::new(0.15132177386470594,0.3335043031018719)    ],
   ///[       Complex::new(0.8593850791476851,0.11354522137267689),   Complex::new(0.4938670101809314,0.7852216948857795),    Complex::new(0.23096625492908301,0.20830882216670715),  Complex::new(0.7400234625821275,0.4639399244725205),    Complex::new(0.5378937891759045,0.7037567515968549),Complex::new(0.2219390019576138,0.23209326206010134),   Complex::new(0.4558367952542631,0.9965121614070889),    Complex::new(0.39631631122312905,0.08633527991619407),  Complex::new(0.9443163419994178,0.42288025964849474),   Complex::new(0.2332370920018188,0.9922786240129193)     ],
   ///[       Complex::new(0.41844631685477907,0.35277367066150905),  Complex::new(0.3475123609164877,0.7101826289551204),    Complex::new(0.5730063661284887,0.48196832299859105),   Complex::new(0.6143248737217993,0.18023274893317168),   Complex::new(0.26770558087500756,0.34859620475408054),      Complex::new(0.6573603929892361,0.09347808620771915),   Complex::new(0.559359670948789,0.7870570717326266),     Complex::new(0.5996313950882705,0.0578118744896874),    Complex::new(0.32955838142838695,0.6696041764734405),   Complex::new(0.21936429159910614,0.4952850734335351)],
   ///[       Complex::new(0.009248256198125528,0.4527930011455513),  Complex::new(0.6849305389546144,0.12102884187585097),   Complex::new(0.4165174797405674,0.9462171528526406),    Complex::new(0.8619475196821003,0.7381751595031852),    Complex::new(0.7596450684070042,0.33049863177437794),       Complex::new(0.37937891021432113,0.0938634467267199),   Complex::new(0.046313554311124834,0.8748186202857586),  Complex::new(0.9142747660514274,0.1720666151092736),    Complex::new(0.1155038542568945,0.8407799452002931),    Complex::new(0.6036564509415651,0.1549954601684234)],
   ///[       Complex::new(0.5280435825255948,0.722128687058149),     Complex::new(0.2958172383395739,0.21101513922732212),   Complex::new(0.6247036944583212,0.2958591539645528),    Complex::new(0.8771245810505679,0.9876277069779977),    Complex::new(0.13778026369279453,0.2925863129303102),       Complex::new(0.6689687280137598,0.24661000255739168),   Complex::new(0.19090555403974846,0.1626089128725561),   Complex::new(0.5354601189785831,0.701258765248206),     Complex::new(0.09332523997739807,0.8382042501840465),   Complex::new(0.12250427701915115,0.7540059801921637)],
   ///[       Complex::new(0.641212141487291,0.12028843766751154),    Complex::new(0.5573689686134607,0.9281730407924285),    Complex::new(0.18228923797612165,0.937966839425133),    Complex::new(0.290229469308307,0.3447329019990513),     Complex::new(0.29421063475530934,0.7783426202185895),       Complex::new(0.2986058932928953,0.8937034310271619),    Complex::new(0.25710953476086523,0.896262417185731),    Complex::new(0.8847389180993006,0.254408582006775),     Complex::new(0.027617968349187068,0.29285850298797206), Complex::new(0.029426361218854565,0.6879945477458963)       ],
   ///[       Complex::new(0.23945753093683192,0.30616733782992506),  Complex::new(0.9881588754627659,0.6754586092988201),    Complex::new(0.6279846571645774,0.07795290055819983),   Complex::new(0.9880650206914865,0.43754117662346503),   Complex::new(0.5668252086231756,0.5654418870268184),Complex::new(0.9563427957776676,0.44960614238550123),   Complex::new(0.8250656417870632,0.5513468135978378),    Complex::new(0.9851555697862651,0.3608225406879005),    Complex::new(0.07324290749628194,0.358150639141774),    Complex::new(0.27138526608582186,0.19393235694918426)   ],
   ///[       Complex::new(0.08899561873929575,0.6885237168549837),   Complex::new(0.6759656028808306,0.16861078919167818),   Complex::new(0.14192833304987176,0.14780523687381544),  Complex::new(0.7793249757513581,0.7530672142302),       Complex::new(0.7685456523065339,0.3380067105229064),Complex::new(0.07062494877030058,0.406943551307159),    Complex::new(0.6391551970178856,0.5412405648127648),    Complex::new(0.5005075046812791,0.3967310304417494),    Complex::new(0.15753551096333432,0.4919072076795531),   Complex::new(0.09696298494836111,0.6188849576238623)    ]];
   /// let schur = Matrix::<_>::schur(&m_complex_f64, 1e-12);
   /// let schur_result = match schur {
   ///    Ok(value) => value,
   ///    Err(e) => {
   ///       println!("Error: {}", e);
   ///       return; // Exit the function early if there's an error
   ///   }
   /// };
   /// schur_result.0.diag().iter().for_each(|x| println!("{:?}",x));
   /// assert!(schur_result.0.diag()[0] == Complex::new(4.863674157977457,4.730078401818301) );
   /// ```
   #[allow(dead_code)]
   pub fn schur(&self, precision: f64) -> Result<(Self,Self),String>
   where 
      T:Clone + Zero + Float + From<f64> + Debug + Signed,
      f64: From<T> + Mul<T>
   {
      assert_eq!(self.rows,self.cols,"Matrix must be a square matrix.");

      let mut mat_a: Matrix<Complex<T>> = self.clone();
      let mut sub_diag = vec![Complex::<T>::zero(); (self.clone().rows() - 1) as usize];
      let mut count_iter = 1;
      //let mut epsilon = Complex::new(T::zero(),T::zero());

      while count_iter < 10000 {

         // let (lambda1,lambda2) = mat_a.sub_matrix(self.rows - 2..=self.rows, self.cols - 2..=self.cols).eigen_2x2();
         // if (lambda1 - mat_a[(self.rows,self.cols)]).magnitude() <= (lambda2 - mat_a[(self.rows,self.cols)]).magnitude() {
         //    epsilon = lambda1;
         // } else{
         //    epsilon = lambda2;
         // }
         //let shift_wilkinson = Matrix::<Complex<T>>::identity(self.rows() as usize).mul(epsilon);
         let (mat_q,mat_r) = Matrix::<Complex<T>>::qr_cht(&mat_a);
         mat_a = mat_r.clone() * mat_q.clone();

         // Compare sub-diagonal for convergence
         if mat_a.skew_diag(-1).iter().zip(sub_diag.clone()).all(|(a,b)| (a.magnitude() - b.magnitude()).abs().to_f64().unwrap() < precision ) == true {
            return Ok((mat_a,mat_q));
         };
         sub_diag = mat_a.skew_diag(-1);
         count_iter += 1;
      }
      Err(String::from("Maximum iteration (10000) exceeded to converge to Schur transform at desired precision"))
   }


   #[allow(dead_code)]
   pub fn eigen_2x2(self) -> (Complex<T>, Complex<T>)
   where
      T: From<f64>,
      f64: From<T>
   {
      // Discriminant (trace^2 - 4 * determinant)
      let trace = self.data[0] + self.data[3];
      let discriminant = (trace * trace) - Complex::<T>::new(<T as From<f64>>::from(4.0), T::zero()) * self.det_2x2();

      // Eigenvalues using the quadratic formula
      let lambda1 = (Complex::from(trace) + discriminant.sqrt()) / Complex::<T>::new(<T as From<f64>>::from(2.0), T::zero());
      let lambda2 = (Complex::from(trace) - discriminant.sqrt()) / Complex::<T>::new(<T as From<f64>>::from(2.0), T::zero());

      (lambda1, lambda2)
   }

}

/// Matrix create macro matrix![[x,y,...],[a,b,...],[c,d,...],...]
#[macro_export]
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
               row_cols = 0;
            }       
         )*
         crum::matrix::Matrix::new(rows, first_row_cols as u128, data)  
      }
   };
   
   ( $rows:expr; $cols:expr; $val:expr) => {{
      Matrix::new($rows, $cols, vec![$val;$rows * $cols])
   }};
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

/// Imiplement multiplication `*` for matrices
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


/// Display a matrix sensibly
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

impl<T: Display + Copy> Debug for Matrix<Complex<T>>
{

   fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
   {
      write!(f, "\n[").expect("Not Written");
      for row_idx in 1..=self.rows{         
         write!(f, "[").expect("Not Written");
         for col_idx in 1..=self.cols {
            if col_idx == self.cols {
               write!(f, "\tComplex::new({},{})", self[(row_idx,col_idx)].real(),self[(row_idx,col_idx)].imag()).expect("Not Written");
            } else {
               write!(f, "\tComplex::new({},{}),", self[(row_idx,col_idx)].real(),self[(row_idx,col_idx)].imag()).expect("Not Written");
            }
         }
         if row_idx == self.rows {
            write!(f, "\t]").expect("Not Written");
         } else {
            write!(f, "\t],\n").expect("Not Written");
         }
         
      }
      write!(f, "]").expect("Not Written");
      Ok(())
   }
}
