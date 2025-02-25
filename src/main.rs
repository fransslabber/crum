use crum::tensor::Tensor;

use std::fs::File;
use std::io::Write;

fn main() -> std::io::Result<()> {

   let a = Tensor::arange(vec![2, 3, 4, 5], 1.0);
   let b = Tensor::arange(vec![5, 4, 3, 2], 1.0);
   let c = a.einsum(&b, "ijkl,mkni");
   println!("c  {}", c);

   let d = a.tensordot(&b, vec![[0,2].to_vec(),[3,1].to_vec()]);
   println!("d  {}", d);
 
   let mut file = File::create("output.txt")?;
   let formatted_text = format!("{c}");
   // Write to file
   file.write_all(formatted_text.as_bytes())?;

   Ok(())
}
