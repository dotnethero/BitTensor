using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

using AttributeName = cudnnBackendAttributeName_t;
using AttributeType = cudnnBackendAttributeType_t;
using DescriptorType = cudnnBackendDescriptorType_t;

public sealed unsafe class CudnnGraph : IDisposable
{
    public cudnnBackendDescriptor_t* Descriptor { get; }

    public CudnnGraph(CudnnContext context, ICudnnOperation[] operations)
    {
        var opCount = operations.Length;
        var opArray = stackalloc cudnnBackendDescriptor_t*[opCount];

        for (var i = 0; i < opCount; i++)
        {
            opArray[i] = operations[i].Descriptor;
        }

        var descriptor = Descriptors.Create(DescriptorType.CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);

        SetHandle(descriptor, context.Handle);
        SetOperations(descriptor, opCount, opArray);
        
        Descriptor = Descriptors.Finalize(descriptor);
    }
    
    private static void SetHandle(cudnnBackendDescriptor_t* descriptor, cudnnContext* context)
    {
        var status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_OPERATIONGRAPH_HANDLE,
            AttributeType.CUDNN_TYPE_HANDLE, 
            elementCount: 1,
            &context);

        Status.EnsureIsSuccess(status);
    }

    private static void SetOperations(cudnnBackendDescriptor_t* descriptor, int count, cudnnBackendDescriptor_t** operations)
    {
        var status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_OPERATIONGRAPH_OPS,
            AttributeType.CUDNN_TYPE_BACKEND_DESCRIPTOR, 
            count,
            operations);
        
        Status.EnsureIsSuccess(status);
    }

    public void Dispose()
    {
        cuDNN.cudnnBackendDestroyDescriptor(Descriptor);
    }
}