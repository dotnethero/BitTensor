using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

using DescriptorType = cudnnBackendDescriptorType_t;
using AttributeName = cudnnBackendAttributeName_t;
using AttributeType = cudnnBackendAttributeType_t;

public sealed unsafe class CudnnTensorDescriptor<T> : IDisposable where T : IFloatingPoint<T>
{
    internal readonly cudnnBackendDescriptor_t* Descriptor;
    internal readonly long* Extents;
    internal readonly long* Strides;

    public CudnnTensorDescriptor(long id, Shape anyShape, long alignment = 4)
    {
        var type = Types.GetDataType<T>();
        var shape = MakeAtLeast2D(anyShape);

        Extents = CudaArray.AllocateAtHost<long>(shape.Dimensions);
        Strides = CudaArray.AllocateAtHost<long>(shape.Dimensions);

        for (var i = 0; i < shape.Dimensions; ++i)
        {
            Extents[i] = shape.Extents[i];
            Strides[i] = shape.Strides[i];
        }

        cudnnStatus_t status;
        cudnnBackendDescriptor_t* descriptor;

        status = cuDNN.cudnnBackendCreateDescriptor(DescriptorType.CUDNN_BACKEND_TENSOR_DESCRIPTOR, &descriptor);
        Status.EnsureIsSuccess(status);

        status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_TENSOR_DATA_TYPE,
            AttributeType.CUDNN_TYPE_DATA_TYPE,
            elementCount: 1,
            &type);

        Status.EnsureIsSuccess(status);

        status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_TENSOR_DIMENSIONS,
            AttributeType.CUDNN_TYPE_INT64,
            elementCount: shape.Dimensions,
            Extents);

        Status.EnsureIsSuccess(status);

        status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_TENSOR_STRIDES,
            AttributeType.CUDNN_TYPE_INT64,
            elementCount: shape.Dimensions,
            Strides);

        Status.EnsureIsSuccess(status);

        status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_TENSOR_UNIQUE_ID,
            AttributeType.CUDNN_TYPE_INT64,
            elementCount: 1,
            &id);

        Status.EnsureIsSuccess(status);

        status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT,
            AttributeType.CUDNN_TYPE_INT64,
            elementCount: 1,
            &alignment);

        Status.EnsureIsSuccess(status);

        status = cuDNN.cudnnBackendFinalize(descriptor);
        Status.EnsureIsSuccess(status);

        Descriptor = descriptor;
    }

    private static Shape MakeAtLeast2D(Shape shape)
    {
        var dimensions = shape.Dimensions < 2 ? 2 : shape.Dimensions;
        var additional = dimensions - shape.Dimensions;
        var ones = Enumerable.Repeat(1, additional);
        return [..ones, ..shape];
    }
    public void Dispose()
    {
        cudnnStatus_t status;

        status = cuDNN.cudnnBackendDestroyDescriptor(Descriptor);
        Status.EnsureIsSuccess(status);

        CudaArray.FreeHost(Extents);
        CudaArray.FreeHost(Strides);
    }
}
