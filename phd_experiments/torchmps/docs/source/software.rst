Related Libraries
#################

There are plenty of other software libraries that let you work with matrix product states/tensor trains. The following is a (non-exhaustive!) list:

`ITensor <https://itensor.org/>`_ (Julia, C++): General-purpose library for working with tensor networks over arbitrary graphs. ITensor is `broadly used <https://itensor.org/docs.cgi?page=papers>`_ in the many-body physics community, with more information available `here <https://arxiv.org/abs/2007.14822>`_.

`TensorNetwork <https://github.com/google/TensorNetwork>`_ (Python): Newer library for working with tensor networks over arbitrary graph structures. Can be used with JAX, TensorFlow, PyTorch or NumPy as a backend, more information available `here <https://arxiv.org/abs/1905.01330>`_.

`T3F <https://github.com/Bihaqo/t3f>`_ (Python): TensorFlow library with lots of support for tensor train factorizations of matrices, for example as used to `tensorize neural networks <https://arxiv.org/abs/1509.06569>`_. Also supports Riemannian optimization.

`quimb <https://github.com/jcmgray/quimb>`_ (Python): Quantum information and many-body library which includes tensor network functionality. Can be used with TensorFlow, PyTorch, JAX or autograd as backend, more information available `here <https://joss.theoj.org/papers/10.21105/joss.00819>`_.

`tntorch <https://github.com/rballester/tntorch>`_ (Python): PyTorch library implementing many different tensor factorizations, including CP, Tucker, and tensor train.
