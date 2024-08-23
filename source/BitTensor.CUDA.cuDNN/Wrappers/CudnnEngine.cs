using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

using AttributeName = cudnnBackendAttributeName_t;
using AttributeType = cudnnBackendAttributeType_t;
using DescriptorType = cudnnBackendDescriptorType_t;

public sealed unsafe class CudnnEngine : IDisposable
{
    public cudnnBackendDescriptor_t* Descriptor { get; }

    public CudnnEngine(CudnnGraph graph, long globalIndex = 0)
    {
        var descriptor = Descriptors.Create(DescriptorType.CUDNN_BACKEND_ENGINE_DESCRIPTOR);

        SetGraph(descriptor, graph.Descriptor);
        SetGlobalIndex(descriptor, globalIndex);

        Descriptor = Descriptors.Finalize(descriptor);
    }
    
    private static void SetGraph(cudnnBackendDescriptor_t* descriptor, cudnnBackendDescriptor_t* graph)
    {
        var status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_ENGINE_OPERATION_GRAPH,
            AttributeType.CUDNN_TYPE_BACKEND_DESCRIPTOR, 
            elementCount: 1,
            &graph);

        Status.EnsureIsSuccess(status);
    }
    
    private static void SetGlobalIndex(cudnnBackendDescriptor_t* descriptor, long globalIndex)
    {
        var status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_ENGINE_GLOBAL_INDEX,
            AttributeType.CUDNN_TYPE_INT64, 
            elementCount: 1,
            &globalIndex);

        Status.EnsureIsSuccess(status);
    }

    public void Dispose()
    {
        cuDNN.cudnnBackendDestroyDescriptor(Descriptor);
    }
}