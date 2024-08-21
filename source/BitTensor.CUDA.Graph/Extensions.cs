using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Graph;

// ReSharper disable CheckNamespace

namespace BitTensor.CUDA;

public static class Extensions
{
    public static CudaVariable<T> CreateNode<T>(this CudaContext context, Shape shape) 
        where T : unmanaged, IFloatingPoint<T> =>
        new(context, context.Allocate<T>(shape));

    public static CudaVariable<T> CreateNode<T>(this CudaContext context, Shape shape, T[] values)
        where T : unmanaged, IFloatingPoint<T> =>
        new(context, context.Allocate<T>(shape, values));

    public static CudaVariable<T> CreateNode<T>(this CudaContext context, T value)
        where T : unmanaged, IFloatingPoint<T> =>
        new(context, context.Allocate<T>([], [value]));
    
    public static CudaVariable<T> AsNode<T>(this CudaTensor<T> tensor, CudaContext context) 
        where T : unmanaged, IFloatingPoint<T> => 
        new(context, tensor);
}
