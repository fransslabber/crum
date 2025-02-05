use crum::matrix;
use crum::complex::Complex;


fn main() {

   let m_complex_f64 = matrix![[Complex::new(1.0,1.0),Complex::new(2.0,0.0)],
                                                  [Complex::new(3.0,0.0),Complex::new(4.0,-1.0)]];

   let (u,sigma,v) = m_complex_f64.svd_qr(1e-15).unwrap();
   println!("Left Singular Vectors {u}\nSigma {sigma}\nRight Singular Vectors {v}");
   println!("Reconstructed A {}", u * sigma * v.trans().conj());

}
/*Matrix A:
   1.0000 + 1.0000i   2.0000 + 0.0000i
   3.0000 + 0.0000i   4.0000 - 1.0000i

Computed Singular Values:
    5.6289
    0.5618

Pinverse Sigma:
    0.1777         0
         0    1.7800

Left Singular Vectors (U):
   0.3754 + 0.2007i   0.2303 - 0.8751i
   0.9030 + 0.0582i   0.0717 + 0.4196i

Right Singular Vectors (V):
   0.5836 - 0.0000i  -0.7647 + 0.2731i
   0.7647 + 0.2731i   0.5836 + 0.0000i

Reconstructed A (U * Sigma * V') for verification:
   1.0000 + 1.0000i   2.0000 + 0.0000i
   3.0000 + 0.0000i   4.0000 - 1.0000i
 */