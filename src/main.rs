use crum::tensor::{contract, flatpack, Tensor};
use crum::tensor;
use std::ops::RangeInclusive;
use ndarray::{Array4, Array6, Axis};
use std::fs::File;
use std::io::Write;

fn main() -> std::io::Result<()> {

   let a = Tensor::arange(vec![2, 3, 4, 5], 1.0);
   let b = Tensor::arange(vec![5, 4, 3, 2], 1.0);

   // let a = Tensor::arange(vec![2, 2, 2], 1.0);
   // let b = Tensor::arange(vec![2, 2, 2], 9.0);

  
   //println!("a as dim {} {:?}",2, flatpack(&a, 2, 0,0));
   //println!("b as dim {} {:?}",0, flatpack(&b, 0, 0,0));
   let c =  contract(vec![3], &a, vec![0], &b);
      
   let mut file = File::create("output.txt")?;
   let formatted_text = format!("{c}");
   // Write to file
   file.write_all(formatted_text.as_bytes())?;

   Ok(())
}
