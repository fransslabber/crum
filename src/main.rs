use crum::tensor;

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
   println!("{:?}",t3);

   let t2 = t3.subtensor(&vec![1..=1,0..=1,1..=2]).unwrap();
   assert!(t2.shape().iter().all(|x| *x==2));
   assert!(t2[&vec![1,1]] == 12.0 && t2[&vec![1,0]] == 11.0);


   println!("{:?}",t2);
   println!("{}", t2[&vec![0,1]]);

   // [[8.0, 9.0],
   //  [11.0,12.0]]

   // we want to t3[0..-1, 1, 1..=2]; returns a 'sub'-tensor of dim 2 because it has one constant index.
   // so that t3[1,1,1]; returns tensor[1.0] which can be cast to scalar... easily?
    

   // let t12 = tensor![[[[[[[[[[[[[8,4,5]]]]]]]]]]]]];
   // println!("{:?}",t12);

}
/* Output
Tensor { shape: [2, 2, 3], strides: [6, 3, 1], data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0] }
Tensor { shape: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3], strides: [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1], data: [8, 4, 5] }
*/