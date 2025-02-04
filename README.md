# crum
Complex (Numbers) Rust Und Matrices

*crum* is a work-in-progress Rust crate for implementing complex numbers and matrices with a large focus on complex matrices—entirely from scratch, with no unsafe calls to established libraries such as LAPACK. 

NB: This is an experimental in beta package, use at your own discretion. Some functionality, although defined, have not been implemented.

 The repository owner accepts no responsibility nor liability for any consequences of the use of this package.

# Features
## Complex Numbers
- num_traits Float and Num for generic complex number type Complex(incomplete)`<T>`
- mimic std c++ `<complex>` functionality
- generic for all primitive types

## Matrices
- generic to all types implementing num_traits Float and Num
- extra vector functions
- complex number specific functionality such Complex Householder Transform, Complex QR decomposition,
  Schur Decomposition
- LU Decomposition(Gauss Elimination with Partial Pivot) for real matrices.
- Real Square matrix determinant
- Solve linear system of equations with LU decomposition.
