# kevz-svm
This library implements the Pegasos algorithm for efficiently solving the dual 
form of the soft-margin support vector machine. It's my first time working with
Rust so apologies if the code isn't particularly idiomatic.

Also, this code does not work with BLAS bindings yet - it uses for-loops for 
matrix multiplication - so it's performance is terrible.
