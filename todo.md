## Basics

* unit tests for all basic operations - WIP (add, matmul, sum)
* unit tests for all basic gradients - WIP (add, matmul)
* unit tests for second order gradient
* kernels as separate methods! (DONE)
* propagation as separate methods! (HOLD)

## Core

* split to var and computed tensor (DONE)
* variables and lazy mode  (DONE)
  invalidate tree on variable change (DONE)
* reduce number of tensors on reduce axis (DONE)
* expand dimensions operation - WIP (prepend, append)
* slice, expand and transpose gradients (DONE)
* sum tensor array oparation - WIP (DONE) - optimize!
* sum along axis operation (DONE)
* broadcasts for operations - (DONE)
* vector _x_ matrix - (DONE)
* softmax, relu, cross_entropy

## Stages

* train linear module for boolean op (DONE)
* train 2-layer for XOR (DONE)
* train MLP for MNIST, batches
* train CNN for MNIST
* train autoencoder
* train transformer

## Misc

* check out `from jax.test_util import check_grads`
