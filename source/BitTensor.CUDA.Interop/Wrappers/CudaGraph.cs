using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

public sealed unsafe class CudaGraph : IDisposable
{
    internal readonly CUgraph_st* Pointer;

    public CudaGraph()
    {
        CUgraph_st* graph = null;
        
        var status = cudaRT.cudaGraphCreate(&graph, 1); // TODO: Add flags enum
        Status.EnsureIsSuccess(status);
    }
    
    public CudaGraph(CUgraph_st* graph)
    {
        Pointer = graph;
    }

    public CudaGraphInstance CreateInstance()
    {
        CUgraphExec_st* graphInstance;
        
        var status = cudaRT.cudaGraphInstantiate(&graphInstance, Pointer, 1);
        Status.EnsureIsSuccess(status);

        return new(graphInstance);
    }

    public void Dispose()
    {
        cudaRT.cudaGraphDestroy(Pointer);
    }
}