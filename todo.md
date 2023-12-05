## Basics

* unit tests for all basic operations - WIP (matmul)
* unit tests for all basic gradients - WIP (matmul)
* kernels as separate methods! (DONE)
* propagation as separate methods! (HOLD)

## Core

* split to var and computed tensor (DONE)
* variables and lazy mode  (DONE)
  invalidate tree on variable change (DONE)
* reduce number of tensors on reduce axis (DONE)
* expand dimensions operation - WIP (prepend, append)
* slice, expand and transpose gradients
* sum tensor array oparation - WIP (check grad)
* sum along axis operation
* broadcasts for operations - WIP (matmul, add)
* vector _x_ matrix - WIP (need second grad)
* softmax, relu, cross_entropy
* unit tests for second order gradient

## Stages

* train linear module for boolean op (DONE)
* train 2-layer for XOR (DONE)
* train MLP for MNIST, batches
* train CNN for MNIST
* train autoencoder
* train transformer

## Misc

* check out `from jax.test_util import check_grads`
