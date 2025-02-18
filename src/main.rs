use crum::tensor::{contract, flatslice, Tensor};
use crum::tensor;
use std::ops::RangeInclusive;
use ndarray::{Array4, Array6, Axis};
use std::fs::File;
use std::io::Write;

fn main() -> std::io::Result<()> {


   // let a = Tensor::arange(vec![2, 3, 4], 1.0);
   // let b = Tensor::arange(vec![4, 5, 6], 25.0);
   // println!("a  {}",a);
   // println!("b  {}",b);

   // let lh = flatpack(&a,2,0,0);
   // println!("lh size {} {:?}", lh.len(), lh);

   // let rh = flatpack(&b,0,0,0);
   // println!("rh size {} {:?}", rh.len(), rh);


   let a = Tensor::arange(vec![2, 3, 4, 5], 1.0);
   let b = Tensor::arange(vec![5, 4, 3, 2], 1.0);

   let c =  contract(vec![1], &a, vec![2], &b);
      
   let mut file = File::create("output.txt")?;
   let formatted_text = format!("{c}");
   // Write to file
   file.write_all(formatted_text.as_bytes())?;

   Ok(())
}
