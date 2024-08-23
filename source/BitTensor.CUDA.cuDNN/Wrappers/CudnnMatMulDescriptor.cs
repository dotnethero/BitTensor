using System.Numerics;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

using DescriptorType = cudnnBackendDescriptorType_t;
using AttributeName = cudnnBackendAttributeName_t;
using AttributeType = cudnnBackendAttributeType_t;

public sealed unsafe class CudnnMatMulDescriptor<T> : IDisposable where T : IFloatingPoint<T>
{
    internal readonly cudnnBackendDescriptor_t* Descriptor;

    public CudnnMatMulDescriptor(
        CudnnTensorDescriptor<T> a,
        CudnnTensorDescriptor<T> b,
        CudnnTensorDescriptor<T> c)
    {
        fixed (cudnnBackendDescriptor_t**
               ax = &a.Descriptor,
               bx = &b.Descriptor,
               cx = &c.Descriptor)
        {
            var settings = CreateMatMulSettings();
            var operation = CreateMatMulOperation(&settings, ax, bx, cx);

            Descriptor = operation;
        }
    }

    private static cudnnBackendDescriptor_t* CreateMatMulOperation(
        cudnnBackendDescriptor_t** settings,
        cudnnBackendDescriptor_t** a,
        cudnnBackendDescriptor_t** b,
        cudnnBackendDescriptor_t** c)
    {
        cudnnBackendDescriptor_t* descriptor;

        var status = cuDNN.cudnnBackendCreateDescriptor(DescriptorType.CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR, &descriptor);
        Status.EnsureIsSuccess(status);
        
        status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_OPERATION_MATMUL_DESC,
            AttributeType.CUDNN_TYPE_BACKEND_DESCRIPTOR, 
            elementCount: 1,
            settings);

        Status.EnsureIsSuccess(status);

        status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_OPERATION_MATMUL_ADESC,
            AttributeType.CUDNN_TYPE_BACKEND_DESCRIPTOR, 
            elementCount: 1,
            a);

        Status.EnsureIsSuccess(status);

        status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_OPERATION_MATMUL_BDESC,
            AttributeType.CUDNN_TYPE_BACKEND_DESCRIPTOR, 
            elementCount: 1,
            b);
        
        Status.EnsureIsSuccess(status);

        status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_OPERATION_MATMUL_CDESC,
            AttributeType.CUDNN_TYPE_BACKEND_DESCRIPTOR, 
            elementCount: 1,
            c);

        Status.EnsureIsSuccess(status);
        
        status = cuDNN.cudnnBackendFinalize(descriptor);
        Status.EnsureIsSuccess(status);

        return descriptor;
    }

    private static cudnnBackendDescriptor_t* CreateMatMulSettings()
    {
        cudnnBackendDescriptor_t* descriptor;

        var type = Types.GetDataType<T>();
        var status = cuDNN.cudnnBackendCreateDescriptor(DescriptorType.CUDNN_BACKEND_MATMUL_DESCRIPTOR, &descriptor);

        Status.EnsureIsSuccess(status);
        
        status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_MATMUL_COMP_TYPE,
            AttributeType.CUDNN_TYPE_DATA_TYPE, 
            elementCount: 1,
            &type);

        Status.EnsureIsSuccess(status);

        status = cuDNN.cudnnBackendFinalize(descriptor);
        Status.EnsureIsSuccess(status);

        return descriptor;
    }

    public void Dispose()
    {
        cudnnStatus_t status;

        status = cuDNN.cudnnBackendDestroyDescriptor(Descriptor);
        Status.EnsureIsSuccess(status);
    }
}