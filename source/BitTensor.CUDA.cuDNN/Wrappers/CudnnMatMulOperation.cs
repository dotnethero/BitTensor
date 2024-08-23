using System.Numerics;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

using DescriptorType = cudnnBackendDescriptorType_t;
using AttributeName = cudnnBackendAttributeName_t;
using AttributeType = cudnnBackendAttributeType_t;

public sealed unsafe class CudnnMatMulOperation<T> : ICudnnOperation where T : unmanaged, IFloatingPoint<T>
{
    public cudnnBackendDescriptor_t* Descriptor { get; }

    public CudnnMatMulOperation(
        CudnnMatMulOperator<T> matmul,
        CudnnTensorDescriptor<T> a,
        CudnnTensorDescriptor<T> b,
        CudnnTensorDescriptor<T> c)
    {
        var descriptor = Descriptors.Create(DescriptorType.CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR);

        SetOperation(descriptor, matmul.Descriptor);
        SetA(descriptor, a.Descriptor);
        SetB(descriptor, b.Descriptor);
        SetC(descriptor, c.Descriptor);

        Descriptor = Descriptors.Finalize(descriptor);
    }

    private static void SetOperation(cudnnBackendDescriptor_t* descriptor, cudnnBackendDescriptor_t* config)
    {
        var status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_OPERATION_MATMUL_DESC,
            AttributeType.CUDNN_TYPE_BACKEND_DESCRIPTOR, 
            elementCount: 1,
            &config);

        Status.EnsureIsSuccess(status);
    }

    private static void SetA(cudnnBackendDescriptor_t* descriptor, cudnnBackendDescriptor_t* tensor)
    {
        var status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_OPERATION_MATMUL_ADESC,
            AttributeType.CUDNN_TYPE_BACKEND_DESCRIPTOR, 
            elementCount: 1,
            &tensor);

        Status.EnsureIsSuccess(status);
    }

    private static void SetB(cudnnBackendDescriptor_t* descriptor, cudnnBackendDescriptor_t* tensor)
    {
        var status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_OPERATION_MATMUL_BDESC,
            AttributeType.CUDNN_TYPE_BACKEND_DESCRIPTOR, 
            elementCount: 1,
            &tensor);
        
        Status.EnsureIsSuccess(status);
    }

    private static void SetC(cudnnBackendDescriptor_t* descriptor, cudnnBackendDescriptor_t* tensor)
    {
        var status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_OPERATION_MATMUL_CDESC,
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