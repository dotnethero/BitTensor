using System.Numerics;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

using DescriptorType = cudnnBackendDescriptorType_t;
using AttributeName = cudnnBackendAttributeName_t;
using AttributeType = cudnnBackendAttributeType_t;

internal sealed unsafe class CudnnReductionOperation<T> : ICudnnOperation where T : unmanaged, IFloatingPoint<T>
{
    public cudnnBackendDescriptor_t* Descriptor { get; }
    
    public CudnnReductionOperation(
        CudnnReductionOperator<T> op,
        CudnnTensorDescriptor<T> x,
        CudnnTensorDescriptor<T> y)
    {
        var descriptor = Descriptors.Create(DescriptorType.CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR);

        SetOperation(descriptor, op.Descriptor);
        SetX(descriptor, x.Descriptor);
        SetY(descriptor, y.Descriptor);

        Descriptor = Descriptors.Finalize(descriptor);
    }

    private static void SetOperation(cudnnBackendDescriptor_t* descriptor, cudnnBackendDescriptor_t* config)
    {
        var status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_OPERATION_REDUCTION_DESC,
            AttributeType.CUDNN_TYPE_BACKEND_DESCRIPTOR, 
            elementCount: 1,
            &config);

        Status.EnsureIsSuccess(status);
    }

    private static void SetX(cudnnBackendDescriptor_t* descriptor, cudnnBackendDescriptor_t* tensor)
    {
        var status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_OPERATION_REDUCTION_XDESC,
            AttributeType.CUDNN_TYPE_BACKEND_DESCRIPTOR, 
            elementCount: 1,
            &tensor);

        Status.EnsureIsSuccess(status);
    }

    private static void SetY(cudnnBackendDescriptor_t* descriptor, cudnnBackendDescriptor_t* tensor)
    {
        var status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_OPERATION_REDUCTION_YDESC,
            AttributeType.CUDNN_TYPE_BACKEND_DESCRIPTOR, 
            elementCount: 1,
            &tensor);

        Status.EnsureIsSuccess(status);
    }

    public void Dispose()
    {
        cuDNN.cudnnBackendDestroyDescriptor(Descriptor);
    }
}