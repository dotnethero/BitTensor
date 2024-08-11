using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Wrappers;
using ILGPU;
using ILGPU.Runtime.Cuda;

// ReSharper disable ConvertToAutoProperty
// ReSharper disable InconsistentNaming

namespace BitTensor.CUDA;

public interface IHasContext
{
    public CuContext GetContext();
}

public partial class CuContext : IDisposable
{
    private readonly Context cuContext;
    private readonly CudaAccelerator cuAccelerator;
    private readonly CuRandContext cuRAND;
    private readonly CuTensorContext cuTENSOR;

    public CuRandContext Random => cuRAND;

    public static CuContext CreateDefault()
    {
        var ctx = Context.CreateDefault();
        var acc = ctx.CreateCudaAccelerator(0);
        return new CuContext(ctx, acc);
    }

    private CuContext(Context ctx, CudaAccelerator acc)
    {
        cuContext = ctx;
        cuAccelerator = acc;
        cuRAND = new CuRandContext(this);
        cuTENSOR = new CuTensorContext();
    }

    public CuTensor<T> Allocate<T>(Shape shape) where T : unmanaged
    {
        var array = cuAccelerator.Allocate<T>(shape.ArraySize);
        return new(this, array, shape);
    }
    
    public CuTensor<T> Allocate<T>(Shape shape, T[] values) where T : unmanaged
    {
        var array = cuAccelerator.Allocate<T>(values);
        return new(this, array, shape);
    }

    public CuTensor<T> AllocateOne<T>()
        where T : unmanaged, INumberBase<T> =>
        Allocate<T>([], [T.One]);

    public void Dispose()
    {
        cuTENSOR.Dispose();
        cuAccelerator.Dispose();
        cuContext.Dispose();
    }
}