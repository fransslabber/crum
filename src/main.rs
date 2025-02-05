use crum::matrix::Matrix;
use crum::complex::Complex;
use crum::matrix;
use crum::tensor;
use crum::tensor::{TensorCoord,Tensor};
use rand::distributions::uniform::SampleUniform;
use rand::Rng;

fn main() {
   let dimensions = vec![3 as usize,6,4,9];

   let mut rng = rand::thread_rng(); // Create a thread-local RNG
   let size = 3*6*4*9;

   // Fill the vector with random numbers in defined range
   let random_numbers:Vec<f64> = (0..size)
      .map(|_| rng.gen_range(0.1..1.0) )
      .collect();



   let t:Tensor<f64> = Tensor::new(dimensions,&random_numbers );
   println!("{:?}",t[vec![2,5,2,7]])
}