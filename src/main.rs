use crum::tensor::Tensor;
use crum::tensor;
use std::ops::RangeInclusive;

fn main() {

   // let bigballs = Tensor::random(vec![5,3,7,10], 0.1..1.0);
   // println!("{}",bigballs);

   // // let littleballs = bigballs.subtensor(&vec![2..=3,2..=2,2..=5,0..=20]).unwrap();
   // // println!("{:?}",littleballs.shape());
   
   // let t3 = tensor!([
   //                                  [
   //                                     [1.0, 2.0, 3.0],
   //                                     [4.0, 5.0, 6.0]
   //                                  ],
   //                                  [
   //                                     [7.0, 8.0, 9.0],
   //                                     [10.0, 11.0, 12.0]
   //                                  ]
   //                               ]);
   // println!("{}",t3);

   // let t2 = t3.subtensor(&vec![1..=1,0..=1,1..=2]).unwrap();
   // assert!(t2.shape().iter().all(|x| *x==2));
   // assert!(t2[&vec![1,1]] == 12.0 && t2[&vec![1,0]] == 11.0);


   // println!("{:?}",t2);
   // println!("{}", t2[&vec![0,1]]);

   // [[8.0, 9.0],
   //  [11.0,12.0]]

   // we want to t3[0..-1, 1, 1..=2]; returns a 'sub'-tensor of dim 2 because it has one constant index.
   // so that t3[1,1,1]; returns tensor[1.0] which can be cast to scalar... easily?
    

   // let t12 = tensor![[[[[[[[[[[[[8,4,5]]]]]]]]]]]]];
   // println!("{:?}",t12);


   // Tensor Contraction
   // let t4 = Tensor::arange(vec!(3,4,2), 1.0);
   // println!("{}",t4);
   //println!("{}",t4.subtensor(&vec!(1..=1,2..=2,0..=1)).unwrap());

   // let t3 = tensor!([
   //                                  [
   //                                     [1.0, 2.0, 3.0],
   //                                     [4.0, 5.0, 6.0]
   //                                  ],
   //                                  [
   //                                     [7.0, 8.0, 9.0],
   //                                     [10.0, 11.0, 12.0]
   //                                  ]
   //                               ]);
   // println!("{}",t3);

   // println!("{}", t4 * t3);

   let m2 = Tensor::arange(vec!(3,4), 1.0);
   println!("{}",m2);
   let m2_1 = Tensor::arange(vec!(4,5), 1.0);
   println!("{}",m2_1);
   println!("{}", m2 * m2_1);

   // let a = tensor!([
   //      [1.0, 2.0, 3.0], 
   //      [4.0, 5.0, 6.0]
   //  ]); // Shape: (2,3)

   //  // Define a 3D tensor B of shape (3,4,2)
   //  let b = tensor!([
   //      [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],  // j = 0
   //      [[9.0, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]],  // j = 1
   //      [[17.0, 18.0], [19.0, 20.0], [21.0, 22.0], [23.0, 24.0]]  // j = 2
   //  ]); // Shape: (3,4,2)

   //  // Perform tensor contraction over axis 1 of A and axis 0 of B
   //  println!("{}", a * b); // Resulting shape: (2,4,2)



}
/* Output
Tensor { shape: [2, 2, 3], strides: [6, 3, 1], data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0] }
Tensor { shape: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3], strides: [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1], data: [8, 4, 5] }
*/