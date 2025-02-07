use crum::tensor::{TCoord,Tensor};
use crum::tensor;

fn main() {
   
   let mut t3 = tensor![[[[[[[[[[[[8]]]]]]]]]]]];
   //    [
   //          [1.0, 2.0, 3.0],
   //          [4.0, 5.0, 6.0]
   //    ],
   //    [
   //          [7.0, 8.0, 9.0],
   //          [10.0, 11.0, 12.0]
   //    ]
   // ]];
   println!("{:?}",t3);

   // let s = vec![0,0,0];
   // println!("{:?}",t3[&s]);
   // t3[&s] = 9.0;
   // println!("{:?}",t3[&s]);

}