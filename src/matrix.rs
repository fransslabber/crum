use crate::complex::Complex;

use std::ops::{Add, Div, Index, IndexMut, Mul, RangeInclusive, Sub};
use std::vec::Vec;
use std::fmt::{Debug, Display};
use num_traits::{Float, One, Zero};
use rand::distributions::uniform::SampleUniform;
use rand::Rng;

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
#[derive(Debug, Clone)]
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

   /// Construct a n x n identity matrix
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

   /// Check if matrix is identity matrix
   #[allow(dead_code)]
   pub fn is_identity(&self) -> bool
   where 
      T: One + PartialEq
   {
      (self.diag()).iter().all(|&x| x == T::one())
   }

   /// Set nth row of a matrix
   #[allow(dead_code)]
   pub fn row_set(self, idx: u128, row: Vec<T>) -> Self
   where 
      {

      let mut data= self.data;
      data.splice((self.cols *(idx - 1)) as usize..((self.cols *(idx - 1))+self.cols) as usize,row);      
 
      Self {
         rows: self.rows,
         cols: self.cols,
         data: data.to_vec()
      }
   }   

   /// Set nth col of a matrix
   #[allow(dead_code)]
   pub fn col_set(self, idx: u128, col: Vec<T>) -> Self
   where 
      {
      let mut data= self.data;      
      let vec_iter = data
      .iter_mut()  // mutable iterator over the vector
      .skip((idx - 1) as usize)  // Skip the first elements
      .step_by(self.cols as usize);
      
      let mut index:usize = 0;
      for x in vec_iter{
         *x = col[index].clone();
         index += 1;
      }

      Self {
         rows: self.rows,
         cols: self.cols,
         data: data.to_vec()
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

}



// impl<T> Matrix<T> {
//    // QR Decomposition - Gram-Schmidt
//    pub fn qr_decomp_gs(&self) ->(Self,Self)
//       where
//          T:Copy + Zero + Float
//          {
//             let mut q = Matrix::new(self.rows, self.cols, vec![T::zero(); (self.rows*self.cols) as usize ]);
//             let mut r = Matrix::new(self.rows, self.cols, vec![T::zero(); (self.rows*self.cols) as usize ]);
//             // For each column in self
//             // Define vector as a n x 1 matrix
//             for i in 1..=self.cols {
//                let mut col_i = self.col(i);

//                // Orthogonalize the current column against all preceding columns
//                for j in 1..=i {
//                   let col_j = self.col(j);
//                   let r_ji = dot_product(&col_j, &col_i);

//                   r[(j,i)] = r_ji; 
                  
//                   col_i = vector_sub(&col_j,&scalar_mul(&col_j,r_ji));

//                }

//                // Normalize column
//                let norm = magnitude(&col_i);
//                r[(i,i)] = norm; 
//                col_i = scalar_div(&col_i,norm);
            
//                // set as ith column of Q matrix
//                q = q.clone().col_set(i, col_i);
//             }
//             (q,r)
//          }

// }


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
   /// assert_eq!(result2, Complex::new(76.0, -42.0));
   /// ```
   pub fn norm_2(vec: &Vec<Complex<T>>) -> f64
   where
      T: Float + From<f64>,
      f64: From<T>
   {
      let sum = vec.iter().fold(0.0, |acc, x| acc + f64::from((x.real() * x.real()) + (x.imag() * x.imag())));
      sum.sqrt()
   }

   /// Frobenius norm of a complex matrix
   ///    
   /// ```
   /// use crum::matrix::Matrix;
   /// use crum::complex::Complex;
   /// let result2 = Matrix::<Complex<f64>>::norm_2(&vec![Complex::new(5.0, 3.0),Complex::new(2.0, 4.0),Complex::new(7.0, 1.0),Complex::new(9.0, 5.0)]);
   /// assert_eq!(result2, Complex::new(76.0, -42.0));
   /// ```
   pub fn norm_frobenius(&self) -> f64
   where
      T: Float + From<f64>,
      f64: From<T>
   {
      let sum = self.data.iter().fold(0.0, |acc, x| acc + f64::from((x.real() * x.real()) + (x.imag() * x.imag())));
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
   
   // Get the complex conjugate of a complex matrix
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
   pub fn qr_cht(mat: Matrix<Complex<T>>) -> (Self,Self)
   where
   T:Copy + Zero + Float + From<f64>,
   f64: From<T> + Mul<T>
   {
      assert!(mat.rows >= mat.cols, "CHT for QR only valid when rows >= cols.");
      // Column by column build the Q and R matrices of A such
      // that Q R = A and Q has orthonormal column vectors 
      // and R is an upper diagonal matrix

      // For a given matrix A in the iteration;
      // Calc CHT for the first col vector
      let mut mat_a = mat.clone();
      let mut mat_cht = Matrix::<Complex<T>>::householder_transform(mat_a.col(1));
      let mut mat_r = mat_cht.clone(); // first time does not need a resize.
      let mut mat_q = mat_cht.clone().trans().conj(); // complex conj transpose?

      let mut start_index = 1;      
      let cycles = (mat.rows-1).min(mat.cols);

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
      (mat_q,mat_r*mat)
   }

   /// Householder Transform for Complex Matrices (CHT)
   pub fn householder_transform(x: Vec<Complex<T>>) -> Self
      where
         T:Copy + Zero + Float + From<f64>,
         f64: From<T> + Mul<T>
         {
            //println!("x {:?}", x);
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

   
   /// Perform Schur decomposition on complex matrix
   #[allow(dead_code)]
   pub fn schur(self, threshold: T) -> Self
   where 
      T:Clone + Zero + Float + From<f64> + Debug,
      f64: From<T> + Mul<T>
   {
       // explicitly shifted QR algo with Rayleigh quotient shift
      let mut mat_a: Matrix<Complex<T>> = self;

      let (mut mat_q,mut mat_r) = Matrix::<Complex<T>>::qr_cht(mat_a.clone());

      let mut_residual = mat_a.clone() - (mat_q.clone() * mat_r.clone() * mat_q.clone().conj().trans());

      // update a
      mat_a = mat_r.clone() * mat_q.clone();
      let mut max_iter = 1;
      let mut epsilon = Matrix::<Complex<T>>::norm_2(&mat_a.skew_diag(-1)).magnitude();


      while epsilon > threshold && max_iter < 1000 {
         epsilon = Matrix::<Complex<T>>::norm_2(&mat_a.skew_diag(-1)).magnitude();
         println!("Epsilon: {:?} Threshold: {:?} Iter: {}", epsilon,threshold,max_iter);
         // start iteration
         //shift = mat_e.clone().trans() * mat_a.clone() * mat_e.clone();
         //intshift = shift[(1,1)];
      // mat_a = (mat_a - mat_i.clone().mul(intshift)).unwrap();      
         (mat_q,mat_r) = Matrix::<Complex<T>>::qr_cht(mat_a);
         //mat_a = ((R * Q) + mat_i.clone().mul(intshift)).unwrap();
         mat_a = mat_r.clone() * mat_q.clone();
         //println!("eval {}", mat_a)     ;
         // end iteration
         max_iter += 1;
      }

      mat_a
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

   /// Generalized eigenvalues - takes a complex Schur decomposition
   /// and finds all real and complex eigenvalues
   #[allow(dead_code)]
   pub fn eigen_schur(&self) -> (Vec<T>,Vec<Complex<T>>)
   where 
      T: Float + From<f64>,
      f64: From<T> + Mul<T>
   {
      assert_eq!(self.rows,self.cols,"Matrix must be a square matrix");
      let mut eigen_values_real = Vec::<T>::new();
      let mut eigen_values_complex = Vec::<Complex<T>>::new();


      // set real-complex split threshold
      let threshold = Complex::new(<T as From<f64>>::from(0.001),<T as From<f64>>::from(0.001)).magnitude();

      // Process diagonal : if block then complex conjugate eigen pair, if not then real eigen value
      // For each elem(ij), if elem(i+1,j) == zero, then elem(ij) is a real eigen value, and skip 1 diag elem, else;
      // the 2x2 sub-matrix i..i+1 x j..j+1 has complex conjugate eigenvalue pairs.

      let mut row_idx = 1_u128;
      let mut col_idx = 1_u128;

      while row_idx <= self.rows {
         while col_idx <= self.cols {

            if row_idx == self.rows && col_idx == self.cols {
               //println!("eigen real: {}",schur[(row_idx,col_idx)]);
               eigen_values_real.push(self[(row_idx,col_idx)].real());
               row_idx += 1;
               col_idx += 1;
            } else{
               if self[(row_idx+1,col_idx)].magnitude() <= threshold {
                  //println!("eigen real: {}",schur[(row_idx,col_idx)]);
                  eigen_values_real.push(self[(row_idx,col_idx)].real());
                  row_idx += 1;
                  col_idx += 1;

               } else {
                  let schur_sub = self.sub_matrix(row_idx as u128..= (row_idx+1)  as u128, col_idx as u128..=(col_idx+1) as u128);
                  let (lambda1,lambda2) = Matrix::eigen_2x2(schur_sub);
                  // Check if we have a complex conjugate pair
                  if lambda1.conj() == lambda2 {
                     //println!("eigen complex: {} {}",lambda1,lambda2 );
                     eigen_values_complex.push(lambda1);
                     eigen_values_complex.push(lambda2);
                     row_idx += 2; col_idx += 2;
                  } else {
                     eigen_values_real.push(self[(row_idx,col_idx)].real());
                     row_idx += 1;
                     col_idx += 1;                     
                  }
               }
            }
         }
      }
      (eigen_values_real,eigen_values_complex)
   }

}

/// Matrix create macro matrix![[x,y,...],[a,b,...],[c,d,...],...]
#[macro_export]
#[allow(unused_assignments)]
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

// Implement scalar * matrix multiplication
impl<T> Mul<T> for Matrix<T>
where 
   T: Copy + Mul<Output = T>
{
   type Output = Self;

   fn mul(self, other: T) -> Self {
      Self {
         rows: self.rows,
         cols: self.cols,
         data: self.data.iter().map(|&x| x*other).collect()
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
