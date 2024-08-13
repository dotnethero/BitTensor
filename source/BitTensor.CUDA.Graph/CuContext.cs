using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA.Graph;

public class CuContext : IDisposable
{
    public CuRandContext cuRAND { get; }
    public CuTensorContext cuTENSOR { get; }

    public static CuContext CreateDefault()
    {
        return new CuContext();
    }

    private CuContext()
    {
        cuRAND = new CuRandContext();
        cuTENSOR = new CuTensorContext();
    }
    
    public CuNode<T> CreateNode<T>(Shape shape) 
        where T : unmanaged, IFloatingPoint<T> => 
        new(this, new(shape));
    
    public CuNode<T> CreateNode<T>(Shape shape, T[] values) 
        where T : unmanaged, IFloatingPoint<T> => 
        new(this, new(shape, values));
    
    public CuNode<T> CreateNode<T>(T value) 
        where T : unmanaged, IFloatingPoint<T> => 
        new(this, new([], [value]));

    public CuTensor<T> Allocate<T>(Shape shape) 
        where T : unmanaged => 
        new(shape);

    public CuTensor<T> Allocate<T>(Shape shape, T[] values) 
        where T : unmanaged =>
        new(shape, values);

    public void Dispose()
    {
        cuTENSOR.Dispose();
    }
}