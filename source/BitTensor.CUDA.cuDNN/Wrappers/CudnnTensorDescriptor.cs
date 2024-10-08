﻿using System.Numerics;
using BitTensor.Abstractions;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

using DescriptorType = cudnnBackendDescriptorType_t;
using AttributeName = cudnnBackendAttributeName_t;
using AttributeType = cudnnBackendAttributeType_t;

internal sealed unsafe class CudnnTensorDescriptor<T> : IDisposable where T : unmanaged, IFloatingPoint<T>
{
    internal readonly long* Extents;
    internal readonly long* Strides;

    public cudnnBackendDescriptor_t* Descriptor { get; }

    public CudnnTensorDescriptor(AbstractTensor tensor, int dimensions = 2) : this(tensor.Shape, tensor.Id, dimensions)
    {
    }

    public CudnnTensorDescriptor(Shape anyShape, long id, int dimensions = 2, bool isVirtual = false)
    {
        var alignment = sizeof(T);
        var type = Types.GetDataType<T>();
        var shape = anyShape.Expand(dimensions);

        Extents = CudaArray.AllocateAtHost<long>(shape.Dimensions);
        Strides = CudaArray.AllocateAtHost<long>(shape.Dimensions);

        for (var i = 0; i < shape.Dimensions; ++i)
        {
            Extents[i] = shape.Extents[i];
            Strides[i] = shape.Strides[i];
        }

        var descriptor = Descriptors.Create(DescriptorType.CUDNN_BACKEND_TENSOR_DESCRIPTOR);

        SetDataType(descriptor, type);
        SetExtents(descriptor, shape.Dimensions, Extents);
        SetStrides(descriptor, shape.Dimensions, Strides);
        SetUniqueId(descriptor, id);
        SetAlignment(descriptor, alignment);
        SetIsVirtual(descriptor, isVirtual);

        Descriptor = Descriptors.Finalize(descriptor);
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
    
    private static void SetIsVirtual(cudnnBackendDescriptor_t* descriptor, bool isVirtual)
    {
        var status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_TENSOR_IS_VIRTUAL,
            AttributeType.CUDNN_TYPE_BOOLEAN,
            elementCount: 1,
            &isVirtual);

        Status.EnsureIsSuccess(status);
    }

    public void Dispose()
    {
        cuDNN.cudnnBackendDestroyDescriptor(Descriptor);
        CudaArray.FreeHost(Extents);
        CudaArray.FreeHost(Strides);
    }
}