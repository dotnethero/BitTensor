using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Graph;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA;

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
    
    public CuTensorNode<T> CreateNode<T>(Shape shape) 
        where T : unmanaged, IFloatingPoint<T> => 
        new(this, Allocate<T>(shape));

    public CuTensor<T> Allocate<T>(Shape shape) 
        where T : unmanaged => 
        new(shape);

    public CuTensor<T> Allocate<T>(Shape shape, T[] values) 
        where T : unmanaged =>
        new(shape, values);

    public CuTensor<T> AllocateOne<T>()
        where T : unmanaged, IFloatingPoint<T> =>
        Allocate<T>([], [T.One]);

    public void Dispose()
    {
        cuTENSOR.Dispose();
    }
}