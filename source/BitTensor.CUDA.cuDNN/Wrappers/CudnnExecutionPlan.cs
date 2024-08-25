using System.Numerics;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

using AttributeName = cudnnBackendAttributeName_t;
using AttributeType = cudnnBackendAttributeType_t;
using DescriptorType = cudnnBackendDescriptorType_t;

public sealed unsafe class CudnnExecutionPlan : IDisposable
{
    internal readonly CudnnContext Context;
    internal readonly CudnnEngineHeuristics Heuristics;
    internal readonly CudnnEngineConfiguration Configuration;

    public cudnnBackendDescriptor_t* Descriptor { get; }

    public CudnnExecutionPlan(CudnnContext context, CudnnGraph graph)
    {
        Context = context;
        Heuristics = new CudnnEngineHeuristics(graph);
        Configuration = Heuristics.GetConfiguration();

        // execution plan

        var descriptor = Descriptors.Create(DescriptorType.CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);

        SetHandle(descriptor, context.Handle);
        SetConfig(descriptor, Configuration.Descriptor);

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
    
    public void Execute<T>(CudnnVariantPack<T> pack) where T : unmanaged, IFloatingPoint<T>
    {
        var status = cuDNN.cudnnBackendExecute(
            Context.Handle,
            this.Descriptor,
            pack.Descriptor);

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

        Configuration.Dispose();
        Heuristics.Dispose();
    }
}