using System.Numerics;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

using DescriptorType = cudnnBackendDescriptorType_t;
using AttributeName = cudnnBackendAttributeName_t;
using AttributeType = cudnnBackendAttributeType_t;

internal sealed unsafe class CudnnPointwiseOperation<T> : ICudnnOperation where T : unmanaged, IFloatingPoint<T>
{
    public cudnnBackendDescriptor_t* Descriptor { get; }

    public CudnnPointwiseOperation(
        CudnnPointwiseOperator<T> pw,
        CudnnTensorDescriptor<T> x,
        CudnnTensorDescriptor<T> b,
        CudnnTensorDescriptor<T> y)
    {
        var descriptor = Descriptors.Create(DescriptorType.CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);

        SetOperation(descriptor, pw.Descriptor);
        SetX(descriptor, x.Descriptor);
        SetB(descriptor, b.Descriptor);
        SetY(descriptor, y.Descriptor);

        Descriptor = Descriptors.Finalize(descriptor);
    }
    
    private static void SetOperation(cudnnBackendDescriptor_t* descriptor, cudnnBackendDescriptor_t* config)
    {
        var status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR,
            AttributeType.CUDNN_TYPE_BACKEND_DESCRIPTOR, 
            elementCount: 1,
            &config);

        Status.EnsureIsSuccess(status);
    }

    private static void SetX(cudnnBackendDescriptor_t* descriptor, cudnnBackendDescriptor_t* tensor)
    {
        var status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_OPERATION_POINTWISE_XDESC,
            AttributeType.CUDNN_TYPE_BACKEND_DESCRIPTOR, 
            elementCount: 1,
            &tensor);

        Status.EnsureIsSuccess(status);
    }

    private static void SetB(cudnnBackendDescriptor_t* descriptor, cudnnBackendDescriptor_t* tensor)
    {
        var status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_OPERATION_POINTWISE_BDESC,
            AttributeType.CUDNN_TYPE_BACKEND_DESCRIPTOR, 
            elementCount: 1,
            &tensor);
        
        Status.EnsureIsSuccess(status);
    }

    private static void SetY(cudnnBackendDescriptor_t* descriptor, cudnnBackendDescriptor_t* tensor)
    {
        var status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_OPERATION_POINTWISE_YDESC,
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