using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Graph;

public class CudaContext : IDisposable
{
    public CuRandContext cuRAND { get; }
    public CuTensorContext cuTENSOR { get; }

    public static CudaContext CreateDefault()
    {
        return new CudaContext();
    }

    private CudaContext()
    {
        cuRAND = new CuRandContext();
        cuTENSOR = new CuTensorContext();
    }
    
    public CudaNode<T> CreateNode<T>(Shape shape) 
        where T : unmanaged, IFloatingPoint<T> => 
        new(this, new(shape));
    
    public CudaNode<T> CreateNode<T>(Shape shape, T[] values) 
        where T : unmanaged, IFloatingPoint<T> => 
        new(this, new(shape, values));
    
    public CudaNode<T> CreateNode<T>(T value) 
        where T : unmanaged, IFloatingPoint<T> => 
        new(this, new([], [value]));

    public CudaTensor<T> Allocate<T>(Shape shape) 
        where T : unmanaged => 
        new(shape);

    public CudaTensor<T> Allocate<T>(Shape shape, T[] values) 
        where T : unmanaged =>
        new(shape, values);

    public void Dispose()
    {
        cuTENSOR.Dispose();
    }
}