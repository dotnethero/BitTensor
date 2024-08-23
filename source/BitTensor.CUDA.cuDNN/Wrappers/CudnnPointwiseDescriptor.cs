using System.Numerics;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

using DescriptorType = cudnnBackendDescriptorType_t;
using AttributeName = cudnnBackendAttributeName_t;
using AttributeType = cudnnBackendAttributeType_t;

public sealed unsafe class CudnnPointwiseDescriptor<T> : IDisposable where T : IFloatingPoint<T>
{
    internal readonly cudnnBackendDescriptor_t* Descriptor;

    public CudnnPointwiseDescriptor(
        CudnnTensorDescriptor<T> a,
        CudnnTensorDescriptor<T> b,
        CudnnTensorDescriptor<T> c)
    {
        fixed (cudnnBackendDescriptor_t**
               ax = &a.Descriptor,
               bx = &b.Descriptor,
               cx = &c.Descriptor)
        {
            var settings = CreatePointwiseSettings();
            var operation = CreatePointwiseOperation(&settings, ax, bx, cx);

            Descriptor = operation;
        }
    }

    private static cudnnBackendDescriptor_t* CreatePointwiseOperation(
        cudnnBackendDescriptor_t** settings,
        cudnnBackendDescriptor_t** x,
        cudnnBackendDescriptor_t** b,
        cudnnBackendDescriptor_t** y)
    {
        cudnnBackendDescriptor_t* descriptor;

        var status = cuDNN.cudnnBackendCreateDescriptor(DescriptorType.CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR, &descriptor);
        Status.EnsureIsSuccess(status);
        
        status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR,
            AttributeType.CUDNN_TYPE_BACKEND_DESCRIPTOR, 
            elementCount: 1,
            settings);

        Status.EnsureIsSuccess(status);

        status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_OPERATION_POINTWISE_XDESC,
            AttributeType.CUDNN_TYPE_BACKEND_DESCRIPTOR, 
            elementCount: 1,
            x);

        Status.EnsureIsSuccess(status);

        status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_OPERATION_POINTWISE_BDESC,
            AttributeType.CUDNN_TYPE_BACKEND_DESCRIPTOR, 
            elementCount: 1,
            b);
        
        Status.EnsureIsSuccess(status);

        status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_OPERATION_POINTWISE_YDESC,
            AttributeType.CUDNN_TYPE_BACKEND_DESCRIPTOR, 
            elementCount: 1,
            y);

        Status.EnsureIsSuccess(status);

        status = cuDNN.cudnnBackendFinalize(descriptor);
        Status.EnsureIsSuccess(status);

        return descriptor;
    }

    private static cudnnBackendDescriptor_t* CreatePointwiseSettings()
    {
        cudnnBackendDescriptor_t* descriptor;

        var type = Types.GetDataType<T>();
        var mode = cudnnPointwiseMode_t.CUDNN_POINTWISE_ADD;
        var status = cuDNN.cudnnBackendCreateDescriptor(DescriptorType.CUDNN_BACKEND_POINTWISE_DESCRIPTOR, &descriptor);

        Status.EnsureIsSuccess(status);
        
        status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_POINTWISE_MODE,
            AttributeType.CUDNN_TYPE_POINTWISE_MODE, 
            elementCount: 1,
            &mode);
        
        Status.EnsureIsSuccess(status);
        
        status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_POINTWISE_MATH_PREC,
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