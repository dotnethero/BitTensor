using System.Numerics;
using BitTensor.CUDA.Graph;

// ReSharper disable CheckNamespace

namespace BitTensor.CUDA;

public static class CuTensorExtensions
{
    public static CudaNode<T> AsNode<T>(this CudaTensor<T> tensor, CudaContext context) 
        where T : unmanaged, IFloatingPoint<T> => 
        new(context, tensor);
}
