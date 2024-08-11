using System.Numerics;
using BitTensor.CUDA.Graph;

// ReSharper disable CheckNamespace

namespace BitTensor.CUDA;

public static class CuTensorExtensions
{
    public static CuTensorNode<T> AsNode<T>(this CuTensor<T> tensor) 
        where T : unmanaged, INumberBase<T> => 
        new(tensor);
}
