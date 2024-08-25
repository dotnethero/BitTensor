using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

using AttributeName = cudnnBackendAttributeName_t;
using AttributeType = cudnnBackendAttributeType_t;
using DescriptorType = cudnnBackendDescriptorType_t;

internal sealed unsafe class CudnnEngineConfiguration : IDisposable
{
    public cudnnBackendDescriptor_t* Descriptor { get; }

    public CudnnEngineConfiguration(cudnnBackendDescriptor_t* descriptor)
    {
        Descriptor = descriptor;
    }

    public CudnnEngineConfiguration(CudnnEngine engine)
    {
        var descriptor = Descriptors.Create(DescriptorType.CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);

        SetEngine(descriptor, engine.Descriptor);

        Descriptor = Descriptors.Finalize(descriptor);
    }
    
    private static void SetEngine(cudnnBackendDescriptor_t* descriptor, cudnnBackendDescriptor_t* engine)
    {
        var status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_ENGINECFG_ENGINE,
            AttributeType.CUDNN_TYPE_BACKEND_DESCRIPTOR, 
            elementCount: 1,
            &engine);

        Status.EnsureIsSuccess(status);
    }

    public long GetWorkspaceSize()
    {
        long workspaceSize;

        var status = cuDNN.cudnnBackendGetAttribute(
            Descriptor,
            AttributeName.CUDNN_ATTR_ENGINECFG_WORKSPACE_SIZE,
            AttributeType.CUDNN_TYPE_INT64, 
            requestedElementCount: 1,
            elementCount: null,
            &workspaceSize);

        Status.EnsureIsSuccess(status);

        return workspaceSize;
    }

    public void Dispose()
    {
        cuDNN.cudnnBackendDestroyDescriptor(Descriptor);
    }
}