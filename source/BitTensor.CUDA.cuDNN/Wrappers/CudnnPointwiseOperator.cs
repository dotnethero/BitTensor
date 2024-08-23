using System.Numerics;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

using DescriptorType = cudnnBackendDescriptorType_t;
using AttributeName = cudnnBackendAttributeName_t;
using AttributeType = cudnnBackendAttributeType_t;

public sealed unsafe class CudnnPointwiseOperator<T> : ICudnnOperator where T : unmanaged, IFloatingPoint<T>
{
    public cudnnBackendDescriptor_t* Descriptor { get; }

    public CudnnPointwiseOperator()
    {
        var descriptor = Descriptors.Create(DescriptorType.CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
        var type = Types.GetDataType<T>();

        SetOperation(descriptor, cudnnPointwiseMode_t.CUDNN_POINTWISE_ADD);
        SetPrecision(descriptor, type);
        
        Descriptor = Descriptors.Finalize(descriptor);
    }

    private static void SetOperation(cudnnBackendDescriptor_t* descriptor, cudnnPointwiseMode_t mode)
    {
        var status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_POINTWISE_MODE,
            AttributeType.CUDNN_TYPE_POINTWISE_MODE, 
            elementCount: 1,
            &mode);
        
        Status.EnsureIsSuccess(status);
    }

    private static void SetPrecision(cudnnBackendDescriptor_t* descriptor, cudnnDataType_t type)
    {
        var status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_POINTWISE_MATH_PREC,
            AttributeType.CUDNN_TYPE_DATA_TYPE, 
            elementCount: 1,
            &type);

        Status.EnsureIsSuccess(status);
    }
    
    public void Dispose()
    {
        cuDNN.cudnnBackendDestroyDescriptor(Descriptor);
    }
}