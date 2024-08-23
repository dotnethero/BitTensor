using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

using DescriptorType = cudnnBackendDescriptorType_t;
using AttributeName = cudnnBackendAttributeName_t;
using AttributeType = cudnnBackendAttributeType_t;

public sealed unsafe class CudnnTensorDescriptor<T> : IDisposable where T : unmanaged, IFloatingPoint<T>
{
    internal readonly cudnnBackendDescriptor_t* Descriptor;
    internal readonly long* Extents;
    internal readonly long* Strides;

    public CudnnTensorDescriptor(long id, Shape anyShape)
    {
        var alignment = sizeof(T);
        var type = Types.GetDataType<T>();
        var shape = MakeAtLeast2D(anyShape);

        Extents = CudaArray.AllocateAtHost<long>(shape.Dimensions);
        Strides = CudaArray.AllocateAtHost<long>(shape.Dimensions);

        for (var i = 0; i < shape.Dimensions; ++i)
        {
            Extents[i] = shape.Extents[i];
            Strides[i] = shape.Strides[i];
        }

        var descriptor = CreateDescriptor();

        SetDataType(descriptor, type);
        SetExtents(descriptor, shape.Dimensions, Extents);
        SetStrides(descriptor, shape.Dimensions, Strides);
        SetUniqueId(descriptor, id);
        SetAlignment(descriptor, alignment);
        Finalize(descriptor);

        Descriptor = descriptor;
    }

    private static cudnnBackendDescriptor_t* CreateDescriptor()
    {
        cudnnBackendDescriptor_t* descriptor;
        cudnnStatus_t status = cuDNN.cudnnBackendCreateDescriptor(DescriptorType.CUDNN_BACKEND_TENSOR_DESCRIPTOR, &descriptor);
        Status.EnsureIsSuccess(status);
        return descriptor;
    }
    
    private static void Finalize(cudnnBackendDescriptor_t* descriptor)
    {
        var status = cuDNN.cudnnBackendFinalize(descriptor);
        Status.EnsureIsSuccess(status);
    }

    public static void SetDataType(cudnnBackendDescriptor_t* descriptor, cudnnDataType_t type)
    {
        var status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_TENSOR_DATA_TYPE,
            AttributeType.CUDNN_TYPE_DATA_TYPE,
            elementCount: 1,
            &type);

        Status.EnsureIsSuccess(status);
    }
    
    private static void SetExtents(cudnnBackendDescriptor_t* descriptor, int dimensions, long* extents)
    {
        var status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_TENSOR_DIMENSIONS,
            AttributeType.CUDNN_TYPE_INT64,
            dimensions,
            extents);

        Status.EnsureIsSuccess(status);
    }
    
    private static void SetStrides(cudnnBackendDescriptor_t* descriptor, int dimensions, long* strides)
    {
        var status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_TENSOR_STRIDES,
            AttributeType.CUDNN_TYPE_INT64,
            dimensions,
            strides);

        Status.EnsureIsSuccess(status);
    }
    
    private static void SetUniqueId(cudnnBackendDescriptor_t* descriptor, long id)
    {
        var status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_TENSOR_UNIQUE_ID,
            AttributeType.CUDNN_TYPE_INT64,
            elementCount: 1,
            &id);

        Status.EnsureIsSuccess(status);
    }
    
    private static void SetAlignment(cudnnBackendDescriptor_t* descriptor, long alignment)
    {
        var status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT,
            AttributeType.CUDNN_TYPE_INT64,
            elementCount: 1,
            &alignment);

        Status.EnsureIsSuccess(status);
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
        cuDNN.cudnnBackendDestroyDescriptor(Descriptor);
        CudaArray.FreeHost(Extents);
        CudaArray.FreeHost(Strides);
    }
}