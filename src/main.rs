use crum::tensor::{contract, Tensor};
use crum::tensor;
use std::ops::RangeInclusive;
use ndarray::{Array4, Array6, Axis};
use std::fs::File;
use std::io::Write;

fn main() -> std::io::Result<()> {

   let a = Tensor::arange(vec![2, 3, 4, 5], 1.0);
   println!("a {}",a);
   let b = Tensor::arange(vec![5, 4, 3, 2], 1.0);
   println!("b {}",b);
   let c =  contract(3, &a, 0, &b);
   let d = contract(2, &c, 3, &c);
   
   let mut file = File::create("output.txt")?;
   let formatted_text = format!("{d}");
   // Write to file
   file.write_all(formatted_text.as_bytes())?;

   Ok(())
}
