# crum
Complex (Numbers) Rust Und Matrices

*crum* is a work-in-progress Rust crate for implementing complex numbers and matrices with a large focus on complex matrices—entirely from scratch, with no unsafe calls to established libraries such as LAPACK.

# Features
## Complex Numbers
- num_traits Float and Num for generic complex number type Complex`<T>`
- mimic std c++ `<complex>` functionality
- generic for all primitive types

## Matrices
- generic to all types implementing num_traits Float and Num
- extra vector functions
- complex number specific functionality such Complex Householder Transform, Complex QR decomposition
