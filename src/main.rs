use crum::tensor::{TCoord,Tensor};
use tensor_macro::tensor;

fn main() {
   let t3 = tensor![[
      [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
      ],
      [
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
      ]
   ]];
   
   let dimensions = vec![2 as usize,2,3];
   let mut t = Tensor::<f64>::new(dimensions, &t3 );
   let s = vec![0,0,0];
   println!("{:?}",t[&s]);
   t[&s] = 9.0;
   println!("{:?}",t[&s]);

}