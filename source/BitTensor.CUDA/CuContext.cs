using BitTensor.Abstractions;
using BitTensor.CUDA.Wrappers;

// ReSharper disable InconsistentNaming

namespace BitTensor.CUDA;

public class CuContext : IDisposable
{
    public readonly CuRandContext cuRAND;
    public readonly CuTensorContext cuTENSOR;

    public CuContext()
    {
        cuRAND = new CuRandContext(this);
        cuTENSOR = new CuTensorContext();
    }

    public CuTensor Allocate(Shape shape)
    {
        var array = CuArray.Allocate<float>(shape.ArraySize);
        return new CuTensor(this, array, shape);
    }
    
    public CuTensor Allocate(Shape shape, float[] values)
    {
        var array = CuArray.Allocate<float>(values);
        return new CuTensor(this, array, shape);
    }

    public CuTensor AllocateOne() => Allocate([], [1]);

    public void Dispose()
    {
        cuTENSOR.Dispose();
    }
}