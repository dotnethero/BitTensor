using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

public sealed unsafe class CudaGraphInstance : IDisposable
{
    internal readonly CUgraphExec_st* Pointer;
    
    public CudaGraphInstance(CUgraphExec_st* graphInstance)
    {
        Pointer = graphInstance;
    }

    public void Launch(CudaStream stream)
    {
        cudaRT.cudaGraphLaunch(Pointer, stream.Pointer);
    }

    public void Dispose()
    {
        cudaRT.cudaGraphExecDestroy(Pointer);
    }
}