use crum::matrix;

fn main() {
   // // Example usage with f64
   // let mut c1 = Complex::new(6.0, 4.0);
   // let c2 = Complex::new(3.0, 5.0);
   
   // // (6+4i)mod(3+5i) = -2+2i
   // println!("c1 {:?} c2 {:?} c1 mod c2 {:?}", c1, c2, c1 % c2);
   // // Addition Assign
   // c1 += c2;
   // println!("c1 += c2 {:?}", c1);
   // // Subtraction Assign
   // c1 -= c2;
   // println!("c1 -= c2 {:?}", c1);
   // // Multiplication Assign
   // c1 *= c2;
   // println!("c1 *= c2 {:?}", c1);
   // // Division Assign
   // c1 /= c2;
   // println!("c1 /= c2 {:?}", c1);

   // // Norm
   // println!("sqr|c1| {:?}", c1.norm());

   // // absolute value/modulus/hypotenuse/magnitude
   // println!("|c1| {:?}", c1.hypot());

   // // Real component
   // println!("Real part c1 {:?}", c1.real());

   // // Imaginary component
   // println!("Imaginery part c1 {:?}", c1.imag());

   // // Complex conjugate
   // println!("Complex conjugate c1 {:?}", c1.conj());

   // // Returns a Complex<T> value from polar coords ( angle in radians )
   // println!("Polar to complex number c1 {:?}", Complex::polar(3.0, Complex::degrees_to_radians(45.0)));


   // /*The projection of a complex number is a mathematical operation that maps the complex number to the Riemann sphere, often used in complex analysis. Specifically:

   // For a complex number z=a+biz=a+bi, the projection is defined as:
   // If zz is finite (∣z∣≠∞∣z∣=∞), the projection is zz itself.
   // If zz is infinite (∣z∣=∞∣z∣=∞), the projection maps zz to a "point at infinity." */
   // println!("Projection c1 {:?} to {:?}", c1, c1.proj());

   // // Returns the phase angle (or angular component) of the complex number x, expressed in radians.
   // println!("Phase angle c1 {:?} to {:?} radians", c1, c1.arg());
 
   

   //Use the matrix! macro to create a Matrix<Complex<f64>>
   // let m_complex_f  = matrix![[Complex::new(0.0, 0.0), Complex::new(6.1, -4.0), Complex::new(3.0, -4.0)],
   //                                                 [Complex::new(6.1, 4.0), Complex::new(1.0, 0.0), Complex::new(2.0, -5.0)],
   //                                                 [Complex::new(3.0, 4.0), Complex::new(2.0, 5.0), Complex::new(2.0, 0.0)]];

   // println!("Complex Real {} Diag {:?}", m_complex_f, m_complex_f.diag());

   // let CHT = Matrix::<Complex<f64>>::householder_transform(vec![Complex::new(6.1, 4.0),
   //             Complex::new(1.0, 0.0), Complex::new(2.0, -5.0),Complex::new(3.0, 4.0), Complex::new(2.0, 5.0),
   //             Complex::new(2.0, 0.0)]);
   
   // let X = matrix![[Complex::new(6.1, 4.0)],
   //                                        [ Complex::new(1.0, 0.0)], 
   //                                        [Complex::new(2.0, -5.0)],
   //                                        [Complex::new(3.0, 4.0)], 
   //                                        [Complex::new(2.0, 5.0)], 
   //                                        [Complex::new(2.0, 0.0)]];

   // println!("CHT {} \nX {} \nCHT*X {}",CHT.clone(),X.clone(), CHT*X );


   // let m_complex_f  = matrix![[Complex::new(7.0, 8.9), Complex::new(6.1, -4.0), Complex::new(3.0, -4.0)],
   //                                                 [Complex::new(6.1, 4.0), Complex::new(1.0, 3.0), Complex::new(2.0, -5.0)],
   //                                                 [Complex::new(3.0, 4.0), Complex::new(2.0, 5.0), Complex::new(2.0, 1.2)],
   //                                                 [Complex::new(6.1, 4.0), Complex::new(1.0, 3.0), Complex::new(2.0, -5.0)],
   //                                                 [Complex::new(7.0, 8.9), Complex::new(6.1, -4.0), Complex::new(3.0, -4.0)],
   //                                                 [Complex::new(6.1, 4.0), Complex::new(1.0, 3.0), Complex::new(2.0, -5.0)],
   //                                                 [Complex::new(3.0, 4.0), Complex::new(2.0, 5.0), Complex::new(2.0, 1.2)],
   //                                                 [Complex::new(6.1, 4.0), Complex::new(1.0, 3.0), Complex::new(2.0, -5.0)],
   //                                                 [Complex::new(7.0, 8.9), Complex::new(6.1, -4.0), Complex::new(3.0, -4.0)]];
   
   //println!( "sub-matrix (2..3,2..3) {}", m_complex_f.clone().sub_matrix(2..=3,2..=3));
   
   //println!( "insert (zero)row, insert (unit)col{}", m_complex_f.clone().insert_row(2, vec![ Complex::new(1.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0) ]));
                                               // .insert_col(u128::max_value(), vec![ Complex::new(1.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0),Complex::new(0.0, 0.0) ]));
   
   //println!( "Augment {}", m_complex_f.augment(2));   

   // let (x,y) = Matrix::<Complex<f64>>::qr_cht(m_complex_f.clone());
   // println!( "Q{}\nR{}\nA{}", x.clone(),y.clone(),x.clone()*y.clone());

   // let m_complex_u64 = Matrix::rnd_matrix(10,3,1.0..=101.0);
   // let (x,y) = Matrix::<Complex<f64>>::qr_cht(m_complex_u64.clone());
   // println!( "Original{}\nQ{}\nR{}\nA{}",m_complex_u64, x.clone(),y.clone(),x.clone()*y.clone());

   // let m_identity_4_4 = Matrix::<Complex<f64>>::identity(4);

   // println!("4x4 Complex Identity {} Check {}", m_identity_4_4, m_identity_4_4.is_identity());

   // let m_complex_i  = matrix![[Complex::new(0, 0), Complex::new(6, -4), Complex::new(3, -4)],
   //                                                 [Complex::new(6, 4), Complex::new(1, 0), Complex::new(2, -5)],
   //                                                 [Complex::new(3, 4), Complex::new(2, 5), Complex::new(2, 0)]];
   
   //println!("Complex Integer {}", m_complex_i);

   
   //println!("Is Hermitian {:?}", m_complex_f.clone().is_hermitian());
   //println!("Is Hermitian {:?}", m_complex_i.clone().is_hermitian());
   //let (q,r) = m_complex_f.qr_decomp_gs(); 

   // println!("QR decomposition \n\nQ:\n\n{}\n\nR:\n\n{}",q,r );

   // println!("QR Recombine \n\nQR:\n\n{}",q*r );

   // println!("{:?} \nIndex(2,1) {:?}", m_complex, m_complex[(2,1)]);

   // m_complex[(2,1)] = Complex::new(7.0, 8.0);

   // println!("Index(2,1) {:?}", m_complex[(2,1)]);

   // println!("Row 2 {:?}", m_complex.row(2));

   // println!("Col 2 {:?}", m_complex.col(2));

   // let m = matrix![
   //    [1.0, 2.0],
   //    [3.0, 4.0]
   //    ];

   // let n = matrix![
   //    [1.0, 2.0, 7.0],
   //    [3.0, 4.0, 8.0]
   //    ];

   // let m_mul = m*n.clone();
   // println!("Result: {:?}", m_mul.data());

   //println!("Complex Matrix Multiplication {:?}", m_complex.clone() * Complex::new(6.0, 4.0) + Complex::new(6.0, 4.0));

   // let o = matrix!([[Complex::new(4.6,2.9),Complex::new(6.6,7.3)]]);  // 1x2
   // let p = matrix!([[Complex::new(4.6,2.9)],[Complex::new(6.6,7.3)]]); // 2x1
   // println!("Sim Vector Dot {:?}", o.clone()*p);

   // println!("Scalar multiplication {:?}", n.mul_scalar(&3.0));

   //let mut m_complex_rep: Matrix<Complex<f64>> = matrix!(3;3; Complex::new(6.0, 4.0); Complex::<f64> );

   // let n = matrix![
   //    [1.0, 2.0, 7.0],
   //    [3.0, 4.0, 8.0]];

   // println!("transpose {:?}", n.trans());

   // let m_complex_f64 = matrix![[Complex::new(0.9501,0.0),Complex::new( 0.8913,0.0),Complex::new(0.8214,0.0),Complex::new(0.9218,0.0)],
   //                                                 [Complex::new(0.2311,0.0),Complex::new(0.7621,0.0),Complex::new(0.4447,0.0),Complex::new(0.7382,0.0)],
   //                                                 [Complex::new(0.6068,0.0),Complex::new(0.4565,0.0),Complex::new(0.6154,0.0),Complex::new(0.1763,0.0)],
   //                                                 [Complex::new(0.4860,0.0),Complex::new(0.0185,0.0),Complex::new(0.7919,0.0),Complex::new(0.4057,0.0)]];  

   // let schur_sub = m_complex_f64.schur().sub_matrix(2..=3, 2..=3);
   // println!( "Schur Decomposition Sub {:?}",Matrix::eigen_2x2(schur_sub));

   let m_f64 = matrix![[      0.4130613594781728,      0.06789616787771548,     0.9656690602669977,      0.935185936307158,       0.36917500405507325,     0.928235073753193,       0.22759238061308734,     0.5960510263095428,      0.26386542598969936,     0.6332493168398782      ],
   [       0.243168572454519     ,  0.31261293138410245   ,  0.5252879127056232   ,   0.0330935153674985    ,  0.9753704689278127   ,   0.8360769896786735   ,   0.6548925063235137   ,   0.1468887302027169   ,   0.15578960258556498,     0.08451336130912403     ],
   [       0.41435210063956374   ,  0.2399665922965562    ,  0.9561861714688091   ,   0.697771062293815     ,  0.010276832821779937 ,   0.07041319820424909  ,   0.589651091754667    ,   0.5742129843073421   ,   0.3117164810753857  ,    0.4865256301292692      ],
   [       0.011234519442551607  ,  0.3769729167821171    ,  0.3831601028613202   ,   0.9278273814572322    ,  0.5884363505264488    ,  0.13525442521533695  ,   0.4659138618756341   ,   0.8644592229079726    ,  0.48920799071578347  ,   0.8844850379014413      ],
   [       0.22699699307514926   ,  0.30855453813133443   ,  0.704016327682634    ,   0.9993467239797109    ,  0.5789833380097665   ,   0.920417276804887    ,   0.5016780974268665   ,   0.1965225115934171   ,   0.23901949135357753   ,  0.27956010393374214     ],
   [       0.46588910028585306   ,  0.5638014697165844    ,  0.17953318066016835  ,   0.8848724908721202    ,  0.31679387457452873  ,   0.018462273272746858 ,   0.5401700282679581   ,   0.6200318715242968   ,   0.6500868231349136   ,   0.475028805865377       ],
   [       0.6519277647881355    ,  0.38278755861532476   ,  0.537292424163817    ,   0.6468661089689082    ,  0.34068558363646023  ,   0.17616993251344673  ,   0.5011992281948784   ,   0.0647587238304115   ,   0.46576864282067476  ,   0.7680519057845863      ],
   [       0.554723676204958     ,  0.7331917287295512    ,  0.4119117955980295   ,   0.03440648890443177   ,  0.2875483259617025   ,   0.08849389708374546  ,   0.15523408045389323  ,   0.12164659635666533  ,   0.9991238886878839   ,   0.9744701290327804      ],
   [       0.35148199904388605   ,  0.06203799828471969   ,  0.3318123341812714   ,   0.474823312748173     ,  0.6748846353426651   ,   0.21680233141866695  ,   0.2822951703232011   ,   0.13303486832700243  ,   0.8110022457223178   ,   0.4172961679035855      ],
   [       0.35208686528649485   ,  0.08035514071119178   ,  0.9146101290660322   ,   0.20577219168723354   ,  0.6125322396803093   ,   0.6948480540565597   ,   0.8208429777375439   ,   0.6523677066929491   ,   0.443688274369478    ,   0.21821260167127138     ]];

   //let m_f64 = Matrix::<f64>::rnd_matrix(10, 10, 0.0..=1.0);
   // Convert real matrix to Complex matrix; Perform Complex Schur decomposition; and then get real and complex eigenvalues

   println!("Real matrix: {}\nEigen values: {:?}",m_f64.clone(), m_f64.to_complex().schur(0.8).eigen_schur());



// Eigenvalues of the matrix:
// 4.6295 + 0.0000i

// 0.0099 + 0.9298i
// 0.0099 - 0.9298i

// -0.7426 + 0.0000i

// -0.4668 + 0.3893i
// -0.4668 - 0.3893i


// 0.8076 + 0.0000i
// 0.1987 + 0.0000i
// 0.3598 + 0.0000i
// 0.5198 + 0.0000i


// 4.6295    0.3087    0.1957   -0.1900   -0.2575    0.1278    0.3557    0.3760   -0.2785   -0.5784
// 0    0.0099    0.7167    0.2046    0.5651   -0.4462    0.1944   -0.1040   -0.1051   -0.3455
// 0   -1.2063    0.0099   -0.2200    0.1864   -0.0703   -0.1499   -0.1534    0.2273   -0.2131
// 0         0         0   -0.7426    0.4764   -0.3989    0.0058   -0.1757    0.2748   -0.2558
// 0         0         0         0   -0.4668    0.8837   -0.0952    0.2044    0.1914    0.4259
// 0         0         0         0   -0.1715   -0.4668    0.2165    0.0905    0.1266    0.0937
// 0         0         0         0         0         0    0.8076   -0.0573    0.2702   -0.0073
// 0         0         0         0         0         0         0    0.1987   -0.2561    0.1325
// 0         0         0         0         0         0         0         0    0.3598   -0.1536
// 0         0         0         0         0         0         0         0         0    0.5198

// Eigenvalues of the complex-valued matrix:
//    4.6295 - 0.0000i

//    0.0099 + 0.9298i
//    0.0099 - 0.9298i

//   -0.7426 + 0.0000i

//   -0.4668 + 0.3893i
//   -0.4668 - 0.3893i


//    0.8076 + 0.0000i
//    0.1987 + 0.0000i
//    0.3598 + 0.0000i
//    0.5198 + 0.0000i


}
