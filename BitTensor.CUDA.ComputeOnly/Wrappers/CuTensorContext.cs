using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.ComputeOnly.Wrappers;

internal unsafe class CuTensorContext : IDisposable
{
    internal readonly cutensorHandle* Handle;

    public CuTensorContext()
    {
        cutensorHandle* handle;

        var status = cuTENSOR.cutensorCreate(&handle);
        if (status != cutensorStatus_t.CUTENSOR_STATUS_SUCCESS)
            throw new CuTensorException(status);

        Handle = handle;
    }

    public void Dispose()
    {
        cuTENSOR.cutensorDestroy(Handle);
    }
}