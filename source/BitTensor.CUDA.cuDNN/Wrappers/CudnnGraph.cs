using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

using AttributeName = cudnnBackendAttributeName_t;
using AttributeType = cudnnBackendAttributeType_t;
using DescriptorType = cudnnBackendDescriptorType_t;

internal sealed unsafe class CudnnGraph : ICudnnGraph
{
    internal readonly CudnnContext Context;
    internal readonly ICudnnOperation[] Operations;

    public cudnnBackendDescriptor_t* Descriptor { get; }

    public CudnnGraph(CudnnContext context, ICudnnOperation[] operations)
    {
        Context = context;
        Operations = operations;

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

    public ICudnnPlan GetExecutionPlan() => GetExecutionPlan(cudnnBackendHeurMode_t.CUDNN_HEUR_MODE_INSTANT);

    public ICudnnPlan GetExecutionPlan(cudnnBackendHeurMode_t mode)
    {
        using var heuristics = new CudnnEngineHeuristics(this, mode);
        
        CudnnExecutionPlan plan = null;
        foreach (var configuration in heuristics.GetConfigurations())
        {
            if (!configuration.TryGetEngine(out var engine))
                continue;
            
            try
            {
                plan = new CudnnExecutionPlan(Context, configuration);
                break;
            }
            catch(Exception ex)
            {
                Console.WriteLine(ex);
            }
        }

        if (plan is null)
        {
            throw new InvalidOperationException("Could not find valid configuration");
        }

        return plan;
    }

    public void Dispose()
    {
        cuDNN.cudnnBackendDestroyDescriptor(Descriptor);
    }
}