use crum::matrix;
use crum::complex::Complex;
use crum::matrix::Matrix;

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

   // let m_f64 = matrix![[      0.4130613594781728,      0.06789616787771548,     0.9656690602669977,      0.935185936307158,       0.36917500405507325,     0.928235073753193,       0.22759238061308734,     0.5960510263095428,      0.26386542598969936,     0.6332493168398782      ],
   // [       0.243168572454519     ,  0.31261293138410245   ,  0.5252879127056232   ,   0.0330935153674985    ,  0.9753704689278127   ,   0.8360769896786735   ,   0.6548925063235137   ,   0.1468887302027169   ,   0.15578960258556498,     0.08451336130912403     ],
   // [       0.41435210063956374   ,  0.2399665922965562    ,  0.9561861714688091   ,   0.697771062293815     ,  0.010276832821779937 ,   0.07041319820424909  ,   0.589651091754667    ,   0.5742129843073421   ,   0.3117164810753857  ,    0.4865256301292692      ],
   // [       0.011234519442551607  ,  0.3769729167821171    ,  0.3831601028613202   ,   0.9278273814572322    ,  0.5884363505264488    ,  0.13525442521533695  ,   0.4659138618756341   ,   0.8644592229079726    ,  0.48920799071578347  ,   0.8844850379014413      ],
   // [       0.22699699307514926   ,  0.30855453813133443   ,  0.704016327682634    ,   0.9993467239797109    ,  0.5789833380097665   ,   0.920417276804887    ,   0.5016780974268665   ,   0.1965225115934171   ,   0.23901949135357753   ,  0.27956010393374214     ],
   // [       0.46588910028585306   ,  0.5638014697165844    ,  0.17953318066016835  ,   0.8848724908721202    ,  0.31679387457452873  ,   0.018462273272746858 ,   0.5401700282679581   ,   0.6200318715242968   ,   0.6500868231349136   ,   0.475028805865377       ],
   // [       0.6519277647881355    ,  0.38278755861532476   ,  0.537292424163817    ,   0.6468661089689082    ,  0.34068558363646023  ,   0.17616993251344673  ,   0.5011992281948784   ,   0.0647587238304115   ,   0.46576864282067476  ,   0.7680519057845863      ],
   // [       0.554723676204958     ,  0.7331917287295512    ,  0.4119117955980295   ,   0.03440648890443177   ,  0.2875483259617025   ,   0.08849389708374546  ,   0.15523408045389323  ,   0.12164659635666533  ,   0.9991238886878839   ,   0.9744701290327804      ],
   // [       0.35148199904388605   ,  0.06203799828471969   ,  0.3318123341812714   ,   0.474823312748173     ,  0.6748846353426651   ,   0.21680233141866695  ,   0.2822951703232011   ,   0.13303486832700243  ,   0.8110022457223178   ,   0.4172961679035855      ],
   // [       0.35208686528649485   ,  0.08035514071119178   ,  0.9146101290660322   ,   0.20577219168723354   ,  0.6125322396803093   ,   0.6948480540565597   ,   0.8208429777375439   ,   0.6523677066929491   ,   0.443688274369478    ,   0.21821260167127138     ]];

   //let m_f64 = Matrix::<f64>::rnd_matrix(10, 10, 0.0..=1.0);
   // Convert real matrix to Complex matrix; Perform Complex Schur decomposition; and then get real and complex eigenvalues

   //println!("Real matrix: {}\nEigen values: {:?}",m_f64.clone(), m_f64.to_complex().schur(0.8).eigen_schur());



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


//let m_f64: Matrix::<Complex<f64>> = Matrix::<Complex<f64>>::rnd_complex_matrix(10, 10, 0.0..=1.0);
let m_complex_f64 = matrix![[      Complex::new(0.5833556123799982,0.5690181027600784),    Complex::new(0.6886043600138049,0.674390821408502),     Complex::new(0.24687850063786915,0.5935898903765723),   Complex::new(0.00933456816360523,0.6587484783595824),   Complex::new(0.23512331858462204,0.15986594969605908),      Complex::new(0.3592667599232367,0.044091292164025304),  Complex::new(0.9128331393696729,0.1833852110584138),    Complex::new(0.5180466720472582,0.05333044453605408),   Complex::new(0.26564558002149125,0.24744281386070038),  Complex::new(0.5795439760266531,0.7097323035461603)],
[       Complex::new(0.19355221625091562,0.07443815946182132),  Complex::new(0.38523666576656257,0.6235838654566793),   Complex::new(0.5655998866671316,0.02796381067764698),   Complex::new(0.9478369597737368,0.5061665241108549),    Complex::new(0.9665277542211836,0.6464090293919905),Complex::new(0.8934413145999256,0.9928347855455917),    Complex::new(0.24012630465410162,0.4511339192624414),   Complex::new(0.0795066100131381,0.16804618159775966),   Complex::new(0.7154655006062255,0.27954112740219067),   Complex::new(0.8093795163995636,0.2647562871405445)     ],
[       Complex::new(0.722107408913046,0.3453081577838271),     Complex::new(0.8641742059855633,0.43503554725558835),   Complex::new(0.6576465324620399,0.07852371724975284),   Complex::new(0.38540857835795383,0.5959496185548973),   Complex::new(0.617794405196579,0.7206737924044645),Complex::new(0.5099081560147524,0.8617303081795931),     Complex::new(0.4464823442359144,0.45949602324474303),   Complex::new(0.35752646144365713,0.8983136848274984),   Complex::new(0.6077708116013137,0.6456302985283519),    Complex::new(0.15132177386470594,0.3335043031018719)    ],
[       Complex::new(0.8593850791476851,0.11354522137267689),   Complex::new(0.4938670101809314,0.7852216948857795),    Complex::new(0.23096625492908301,0.20830882216670715),  Complex::new(0.7400234625821275,0.4639399244725205),    Complex::new(0.5378937891759045,0.7037567515968549),Complex::new(0.2219390019576138,0.23209326206010134),   Complex::new(0.4558367952542631,0.9965121614070889),    Complex::new(0.39631631122312905,0.08633527991619407),  Complex::new(0.9443163419994178,0.42288025964849474),   Complex::new(0.2332370920018188,0.9922786240129193)     ],
[       Complex::new(0.41844631685477907,0.35277367066150905),  Complex::new(0.3475123609164877,0.7101826289551204),    Complex::new(0.5730063661284887,0.48196832299859105),   Complex::new(0.6143248737217993,0.18023274893317168),   Complex::new(0.26770558087500756,0.34859620475408054),      Complex::new(0.6573603929892361,0.09347808620771915),   Complex::new(0.559359670948789,0.7870570717326266),     Complex::new(0.5996313950882705,0.0578118744896874),    Complex::new(0.32955838142838695,0.6696041764734405),   Complex::new(0.21936429159910614,0.4952850734335351)],
[       Complex::new(0.009248256198125528,0.4527930011455513),  Complex::new(0.6849305389546144,0.12102884187585097),   Complex::new(0.4165174797405674,0.9462171528526406),    Complex::new(0.8619475196821003,0.7381751595031852),    Complex::new(0.7596450684070042,0.33049863177437794),       Complex::new(0.37937891021432113,0.0938634467267199),   Complex::new(0.046313554311124834,0.8748186202857586),  Complex::new(0.9142747660514274,0.1720666151092736),    Complex::new(0.1155038542568945,0.8407799452002931),    Complex::new(0.6036564509415651,0.1549954601684234)],
[       Complex::new(0.5280435825255948,0.722128687058149),     Complex::new(0.2958172383395739,0.21101513922732212),   Complex::new(0.6247036944583212,0.2958591539645528),    Complex::new(0.8771245810505679,0.9876277069779977),    Complex::new(0.13778026369279453,0.2925863129303102),       Complex::new(0.6689687280137598,0.24661000255739168),   Complex::new(0.19090555403974846,0.1626089128725561),   Complex::new(0.5354601189785831,0.701258765248206),     Complex::new(0.09332523997739807,0.8382042501840465),   Complex::new(0.12250427701915115,0.7540059801921637)],
[       Complex::new(0.641212141487291,0.12028843766751154),    Complex::new(0.5573689686134607,0.9281730407924285),    Complex::new(0.18228923797612165,0.937966839425133),    Complex::new(0.290229469308307,0.3447329019990513),     Complex::new(0.29421063475530934,0.7783426202185895),       Complex::new(0.2986058932928953,0.8937034310271619),    Complex::new(0.25710953476086523,0.896262417185731),    Complex::new(0.8847389180993006,0.254408582006775),     Complex::new(0.027617968349187068,0.29285850298797206), Complex::new(0.029426361218854565,0.6879945477458963)       ],
[       Complex::new(0.23945753093683192,0.30616733782992506),  Complex::new(0.9881588754627659,0.6754586092988201),    Complex::new(0.6279846571645774,0.07795290055819983),   Complex::new(0.9880650206914865,0.43754117662346503),   Complex::new(0.5668252086231756,0.5654418870268184),Complex::new(0.9563427957776676,0.44960614238550123),   Complex::new(0.8250656417870632,0.5513468135978378),    Complex::new(0.9851555697862651,0.3608225406879005),    Complex::new(0.07324290749628194,0.358150639141774),    Complex::new(0.27138526608582186,0.19393235694918426)   ],
[       Complex::new(0.08899561873929575,0.6885237168549837),   Complex::new(0.6759656028808306,0.16861078919167818),   Complex::new(0.14192833304987176,0.14780523687381544),  Complex::new(0.7793249757513581,0.7530672142302),       Complex::new(0.7685456523065339,0.3380067105229064),Complex::new(0.07062494877030058,0.406943551307159),    Complex::new(0.6391551970178856,0.5412405648127648),    Complex::new(0.5005075046812791,0.3967310304417494),    Complex::new(0.15753551096333432,0.4919072076795531),   Complex::new(0.09696298494836111,0.6188849576238623)    ]];

let (q,r) = Matrix::<_>::qr_cht(m_complex_f64);
println!("q:{}\nr:{}", q,r );


}
