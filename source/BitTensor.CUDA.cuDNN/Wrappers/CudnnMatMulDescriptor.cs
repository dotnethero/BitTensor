using System.Numerics;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

public sealed unsafe class CudnnMatMulDescriptor<T> : IDisposable where T : unmanaged, IFloatingPoint<T>
{
    internal readonly cudnnBackendDescriptor_t* Descriptor;

    public CudnnMatMulDescriptor()
    {
        var descriptor = Descriptors.Create(cudnnBackendDescriptorType_t.CUDNN_BACKEND_MATMUL_DESCRIPTOR);
        var type = Types.GetDataType<T>();

        SetPrecision(descriptor, type);

        Descriptor = Descriptors.Finalize(descriptor);
    }

    private static void SetPrecision(cudnnBackendDescriptor_t* descriptor, cudnnDataType_t type)
    {
        var status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            cudnnBackendAttributeName_t.CUDNN_ATTR_MATMUL_COMP_TYPE,
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