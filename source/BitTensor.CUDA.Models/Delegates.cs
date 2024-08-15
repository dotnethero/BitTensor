using System.Numerics;
using BitTensor.CUDA.Graph;

namespace BitTensor.CUDA.Models;

public delegate CudaNode<T> ActivationFunction<T>(
    CudaNode<T> node) 
    where T : unmanaged, IFloatingPoint<T>;

public delegate CudaNode<T> LossFunction<T>(
    CudaNode<T> output,
    CudaNode<T> desired)
    where T : unmanaged, IFloatingPoint<T>;
