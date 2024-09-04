using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

using AttributeName = cudnnBackendAttributeName_t;
using AttributeType = cudnnBackendAttributeType_t;
using DescriptorType = cudnnBackendDescriptorType_t;

internal sealed unsafe class CudnnEngineHeuristics : IDisposable
{
    public cudnnBackendDescriptor_t* Descriptor { get; }

    public CudnnEngineHeuristics(CudnnGraph graph, cudnnBackendHeurMode_t mode = cudnnBackendHeurMode_t.CUDNN_HEUR_MODE_INSTANT)
    {
        var descriptor = Descriptors.Create(DescriptorType.CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);

        SetGraph(descriptor, graph.Descriptor);
        SetMode(descriptor, mode);

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
    
    public CudnnEngineConfiguration[] GetConfigurations()
    {
        var count = GetConfigurationsCount();
        if (count == 0)
            throw new InvalidOperationException("Configuration not found");

        var configs = new cudnnBackendDescriptor_t*[count];
        var result = new CudnnEngineConfiguration[count];

        for (var i = 0; i < count; ++i)
        {
            configs[i] = Descriptors.Create(DescriptorType.CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
        }

        fixed (cudnnBackendDescriptor_t** pointer = configs)
        {
            var status = cuDNN.cudnnBackendGetAttribute(
                Descriptor,
                AttributeName.CUDNN_ATTR_ENGINEHEUR_RESULTS,
                AttributeType.CUDNN_TYPE_BACKEND_DESCRIPTOR, 
                requestedElementCount: count,
                elementCount: null,
                pointer);

            Status.EnsureIsSuccess(status);
        }

        for (var i = 0; i < count; ++i)
        {
            result[i] = new(configs[i]);
        }
        
        return result;
    }

    private long GetConfigurationsCount()
    {
        var count = -1L;
        
        var countStatus = cuDNN.cudnnBackendGetAttribute(
            Descriptor,
            AttributeName.CUDNN_ATTR_ENGINEHEUR_RESULTS,
            AttributeType.CUDNN_TYPE_BACKEND_DESCRIPTOR, 
            requestedElementCount: 0L,
            elementCount: &count,
            null);
        
        Status.EnsureIsSuccess(countStatus);
        return count;
    }

    public void Dispose()
    {
        cuDNN.cudnnBackendDestroyDescriptor(Descriptor);
    }
}