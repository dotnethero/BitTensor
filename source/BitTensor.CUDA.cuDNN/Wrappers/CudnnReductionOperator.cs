using System.Numerics;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

using DescriptorType = cudnnBackendDescriptorType_t;
using AttributeName = cudnnBackendAttributeName_t;
using AttributeType = cudnnBackendAttributeType_t;

internal sealed unsafe class CudnnReductionOperator<T> : ICudnnOperator where T : unmanaged, IFloatingPoint<T>
{
    public cudnnBackendDescriptor_t* Descriptor { get; }

    public CudnnReductionOperator(cudnnReduceTensorOp_t mode)
    {
        var descriptor = Descriptors.Create(DescriptorType.CUDNN_BACKEND_REDUCTION_DESCRIPTOR);
        var type = Types.GetDataType<T>();

        SetOperation(descriptor, mode);
        SetPrecision(descriptor, type);
        
        Descriptor = Descriptors.Finalize(descriptor);
    }
    
    private static void SetOperation(cudnnBackendDescriptor_t* descriptor, cudnnReduceTensorOp_t mode)
    {
        var status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_REDUCTION_OPERATOR,
            AttributeType.CUDNN_TYPE_REDUCTION_OPERATOR_TYPE, 
            elementCount: 1,
            &mode);
        
        Status.EnsureIsSuccess(status);
    }

    private static void SetPrecision(cudnnBackendDescriptor_t* descriptor, cudnnDataType_t type)
    {
        var status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_REDUCTION_COMP_TYPE,
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