using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

using AttributeName = cudnnBackendAttributeName_t;
using AttributeType = cudnnBackendAttributeType_t;
using DescriptorType = cudnnBackendDescriptorType_t;

public sealed unsafe class CudnnExecutionPlan : IDisposable
{
    public cudnnBackendDescriptor_t* Descriptor { get; }

    public CudnnExecutionPlan(CudnnContext context, CudnnEngineConfiguration config)
    {
        var descriptor = Descriptors.Create(DescriptorType.CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);

        SetHandle(descriptor, context.Handle);
        SetConfig(descriptor, config.Descriptor);

        Descriptor = Descriptors.Finalize(descriptor);
    }
    
    private static void SetHandle(cudnnBackendDescriptor_t* descriptor, cudnnContext* context)
    {
        var status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_EXECUTION_PLAN_HANDLE,
            AttributeType.CUDNN_TYPE_HANDLE, 
            elementCount: 1,
            &context);

        Status.EnsureIsSuccess(status);
    }

    private static void SetConfig(cudnnBackendDescriptor_t* descriptor, cudnnBackendDescriptor_t* config)
    {
        var status = cuDNN.cudnnBackendSetAttribute(
            descriptor,
            AttributeName.CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
            AttributeType.CUDNN_TYPE_BACKEND_DESCRIPTOR, 
            elementCount: 1,
            &config);

        Status.EnsureIsSuccess(status);
    }
    
    public long GetWorkspaceSize()
    {
        long workspaceSize;

        var status = cuDNN.cudnnBackendGetAttribute(
            Descriptor,
            AttributeName.CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
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