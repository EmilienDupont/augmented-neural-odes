Similar Software
################

There are plenty of excellent software packages for manipulating matrix product states/tensor trains, some notable ones including the following:

* [TensorNetwork](https://github.com/google/TensorNetwork) (Python, TensorFlow): Powerful library for defining and manipulating general tensor network models, which is described in [Roberts et al. 2019][Rob].
* [T3F](https://github.com/Bihaqo/t3f) (Python, TensorFlow): Useful for general tensor train applications, T3F includes lots of support for working with tensor train factorizations of matrices (as used in [Novikov et al. 2015][Nov]), and supports Riemannian optimization techniques for improved training.
* [tntorch](https://github.com/rballester/tntorch) (Python, Pytorch): Implements many different tensor factorizations, including CP, Tucker, and tensor train.
* [TNML](https://github.com/emstoudenmire/TNML) (C++, ITensor): Implements the DMRG-style training described in [Stoudenmire and Schwab 2016][S&S], which had a major influence on our adaptive training algorithm.

A defining quality of our library is the emphasis on using matrix product states as functional modules which can be easily interchanged with existing neural network components, while still allowing for interesting MPS-specific features.

[Rob]: https://arxiv.org/abs/1905.01330
[S&S]: https://arxiv.org/abs/1605.05775
[NTO]: https://arxiv.org/abs/1605.03795
[Nov]: https://arxiv.org/abs/1509.06569
