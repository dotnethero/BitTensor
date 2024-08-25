using System.Numerics;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

using DescriptorType = cudnnBackendDescriptorType_t;
using AttributeName = cudnnBackendAttributeName_t;
using AttributeType = cudnnBackendAttributeType_t;

internal sealed unsafe class CudnnMatMulOperator<T> : ICudnnOperator where T : unmanaged, IFloatingPoint<T>
{
    public cudnnBackendDescriptor_t* Descriptor { get; }

    public CudnnMatMulOperator()
    {
        var descriptor = Descriptors.Create(DescriptorType.CUDNN_BACKEND_MATMUL_DESCRIPTOR);
        var type = Types.GetDataType<T>();

        SetPrecision(descriptor, type);

        Descriptor = Descriptors.Finalize(descriptor);
    }

    private static void SetPrecision(cudnnBackendDescriptor_t* descriptor, cudnnDataType_t type)
    {
        var status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_MATMUL_COMP_TYPE,
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