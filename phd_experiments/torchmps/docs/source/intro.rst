TorchMPS: Matrix Product States in Pytorch
##########################################

TorchMPS is a framework for working with matrix product state (also known as MPS or tensor train) models within Pytorch. Our MPS models are written as Pytorch Modules, and can simply be viewed as differentiable black boxes that are interchangeable with standard neural network layers. However, the rich structure of MPS's allows for more interesting behavior, such as:

 * A novel adaptive training algorithm (inspired by [Stoudenmire and Schwab 2016][S&S]), which dynamically varies the MPS hyperparameters (MPS bond dimensions) during the course of training.
 * Mechanisms for controlling MPS geometry, such as custom "routing" of the MPS through different regions of the input data, or periodic boundary conditions that give the MPS a circular topology.
 * Choice of tensor contraction strategies, which represent flexible tradeoffs between computational cost and parallelizability of the computation graph.

What our MPS Models Do
======================

The function computed by our MPS Module comes from embedding input data in a high-dimensional feature space before contracting it with an MPS in the same space, as described in [Novikov, Trofimov, and Oseledets 2016][NTO] and [Stoudenmire and Schwab 2016][S&S]. For scalar outputs, this contraction step is formally identical to linear regression, but the (exponentially) large feature space and MPS-parameterized weight vector makes the overall function significantly more expressive. In general, the output is associated with a single site of the MPS, whose placement within the network is a hyperparameter that varies the inductive bias towards different regions of the input data.

