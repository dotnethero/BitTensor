using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Wrappers;

// ReSharper disable ConvertToAutoProperty
// ReSharper disable InconsistentNaming

namespace BitTensor.CUDA;

public interface IHasContext
{
    public CuContext GetContext();
}

public partial class CuContext : IDisposable
{
    private readonly CuRandContext cuRAND;
    private readonly CuTensorContext cuTENSOR;

    public CuRandContext Random => cuRAND;

    public static CuContext CreateDefault()
    {
        return new CuContext();
    }

    private CuContext()
    {
        cuRAND = new CuRandContext(this);
        cuTENSOR = new CuTensorContext();
    }

    public CuTensor<T> Allocate<T>(Shape shape) where T : unmanaged
    {
        var array = CuArray.Allocate<T>(shape.ArraySize);
        return new(this, array, shape);
    }
    
    public CuTensor<T> Allocate<T>(Shape shape, T[] values) where T : unmanaged
    {
        var array = CuArray.Allocate<T>(values);
        return new(this, array, shape);
    }

    public CuTensor<T> AllocateOne<T>()
        where T : unmanaged, INumberBase<T> =>
        Allocate<T>([], [T.One]);

    public void Dispose()
    {
        cuTENSOR.Dispose();
    }
}