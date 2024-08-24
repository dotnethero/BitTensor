using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

using AttributeName = cudnnBackendAttributeName_t;
using AttributeType = cudnnBackendAttributeType_t;
using DescriptorType = cudnnBackendDescriptorType_t;

public sealed unsafe class CudnnEngineHeuristics : IDisposable
{
    public cudnnBackendDescriptor_t* Descriptor { get; }

    public CudnnEngineHeuristics(CudnnGraph graph)
    {
        var descriptor = Descriptors.Create(DescriptorType.CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);

        SetGraph(descriptor, graph.Descriptor);
        SetMode(descriptor, cudnnBackendHeurMode_t.CUDNN_HEUR_MODE_INSTANT);

        Descriptor = Descriptors.Finalize(descriptor);
    }

    private static void SetGraph(cudnnBackendDescriptor_t* descriptor, cudnnBackendDescriptor_t* graph)
    {
        var status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
            AttributeType.CUDNN_TYPE_BACKEND_DESCRIPTOR,
            elementCount: 1,
            &graph);

        Status.EnsureIsSuccess(status);
    }

    private static void SetMode(cudnnBackendDescriptor_t* descriptor, cudnnBackendHeurMode_t mode)
    {
        var status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_ENGINEHEUR_MODE,
            AttributeType.CUDNN_TYPE_HEUR_MODE,
            elementCount: 1,
            &mode);

        Status.EnsureIsSuccess(status);
    }
    
    public CudnnEngineConfiguration GetConfiguration()
    {
        var count = -1L;
        var config = Descriptors.Create(DescriptorType.CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);

        var status = cuDNN.cudnnBackendGetAttribute(
            Descriptor,
            AttributeName.CUDNN_ATTR_ENGINEHEUR_RESULTS,
            AttributeType.CUDNN_TYPE_BACKEND_DESCRIPTOR, 
            requestedElementCount: 1L,
            elementCount: &count,
            &config);

        Status.EnsureIsSuccess(status);

        if (count == -1)
            throw new InvalidOperationException("Configuration not found");

        return new CudnnEngineConfiguration(config);
    }

    public void Dispose()
    {
        cuDNN.cudnnBackendDestroyDescriptor(Descriptor);
    }
}