using System.Numerics;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

public sealed unsafe class CudnnPointwiseDescriptor<T> : IDisposable where T : unmanaged, IFloatingPoint<T>
{
    internal readonly cudnnBackendDescriptor_t* Descriptor;

    public CudnnPointwiseDescriptor()
    {
        var descriptor = Descriptors.Create(cudnnBackendDescriptorType_t.CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
        var type = Types.GetDataType<T>();

        SetOperation(descriptor, cudnnPointwiseMode_t.CUDNN_POINTWISE_ADD);
        SetPrecision(descriptor, type);
        
        Descriptor = Descriptors.Finalize(descriptor);
    }

    private static void SetOperation(cudnnBackendDescriptor_t* descriptor, cudnnPointwiseMode_t mode)
    {
        var status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            cudnnBackendAttributeName_t.CUDNN_ATTR_POINTWISE_MODE,
            cudnnBackendAttributeType_t.CUDNN_TYPE_POINTWISE_MODE, 
            elementCount: 1,
            &mode);
        
        Status.EnsureIsSuccess(status);
    }

    private static void SetPrecision(cudnnBackendDescriptor_t* descriptor, cudnnDataType_t type)
    {
        var status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            cudnnBackendAttributeName_t.CUDNN_ATTR_POINTWISE_MATH_PREC,
            cudnnBackendAttributeType_t.CUDNN_TYPE_DATA_TYPE, 
            elementCount: 1,
            &type);

        Status.EnsureIsSuccess(status);
    }
    
    public void Dispose()
    {
        cuDNN.cudnnBackendDestroyDescriptor(Descriptor);
    }
}