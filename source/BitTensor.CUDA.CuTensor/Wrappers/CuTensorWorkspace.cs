namespace BitTensor.CUDA.Wrappers;

internal sealed unsafe class CuTensorWorkspace : IDisposable
{
    internal readonly void* Pointer = (void*)0;
    internal readonly ulong Bytes;

    public CuTensorWorkspace(ulong bytes)
    {
        Bytes = bytes;

        if (Bytes > 0)
            Pointer = CudaArray.AllocateRaw((uint)bytes);
    }

    public void Dispose()
    {
        if (Bytes > 0)
            CudaArray.Free(Pointer);
    }
}