using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

public sealed unsafe class CudaStream : IDisposable
{
    public static readonly CUstream_st* Default = (CUstream_st*)0;

    public readonly CUstream_st* Pointer;

    public CudaStream()
    {
        CUstream_st* stream = null;

        var status = cudaRT.cudaStreamCreate(&stream);
        Status.EnsureIsSuccess(status);

        Pointer = stream;
    }

    public void Synchronize()
    {
        var status = cudaRT.cudaStreamSynchronize(Pointer);
        Status.EnsureIsSuccess(status);
    }

    public void BeginCapture()
    {
        var status = cudaRT.cudaStreamBeginCapture(Pointer, cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal);
        Status.EnsureIsSuccess(status);
    }

    public CudaGraph EndCapture()
    {
        CUgraph_st* graph = null;

        var status = cudaRT.cudaStreamEndCapture(Pointer, &graph);
        Status.EnsureIsSuccess(status);

        return new(graph);
    }

    public void Dispose()
    {
        cudaRT.cudaStreamDestroy(Pointer);
    }
}