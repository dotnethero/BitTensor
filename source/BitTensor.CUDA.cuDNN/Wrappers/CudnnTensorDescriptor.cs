using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

public sealed unsafe class CudnnTensorDescriptor<T> : IDisposable where T : IFloatingPoint<T>
{
    internal readonly cudnnTensorStruct* Descriptor;
    internal readonly int* Extents;
    internal readonly int* Strides;

    public CudnnTensorDescriptor(Shape anyShape)
    {
        var type = Types.GetDataType<T>();
        var shape = MakeAtLeast4D(anyShape);

        cudnnTensorStruct* descriptor;
        cudnnStatus_t status;

        status = cuDNN.cudnnCreateTensorDescriptor(&descriptor);
        Status.EnsureIsSuccess(status);

        if (shape.Dimensions == 4)
        {
            status = cuDNN.cudnnSetTensor4dDescriptorEx(
                descriptor, type, 
                shape.Extents[^4], // extents
                shape.Extents[^3],
                shape.Extents[^2],
                shape.Extents[^1], 
                shape.Strides[^4], // strides
                shape.Strides[^3],
                shape.Strides[^2],
                shape.Strides[^1]);

            Status.EnsureIsSuccess(status);
        }
        else
        {
            Extents = CudaArray.AllocateAtHost<int>(shape.Dimensions);
            Strides = CudaArray.AllocateAtHost<int>(shape.Dimensions);

            for (var i = 0; i < shape.Dimensions; ++i)
            {
                Extents[i] = shape.Extents[i];
                Strides[i] = shape.Strides[i];
            }

            status = cuDNN.cudnnSetTensorNdDescriptor(descriptor, type, shape.Dimensions, Extents, Strides);
            Status.EnsureIsSuccess(status);
        }

        Descriptor = descriptor;
    }

    private static Shape MakeAtLeast4D(Shape shape)
    {
        var dimensions = shape.Dimensions < 4 ? 4 : shape.Dimensions;
        var additional = dimensions - shape.Dimensions;
        var ones = Enumerable.Repeat(1, additional);
        return [..ones, ..shape];
    }

    public void Dispose()
    {
        cuDNN.cudnnDestroyTensorDescriptor(Descriptor);
        CudaArray.FreeHost(Extents);
        CudaArray.FreeHost(Strides);
    }
}
