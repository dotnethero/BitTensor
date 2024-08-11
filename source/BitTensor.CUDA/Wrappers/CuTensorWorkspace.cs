namespace BitTensor.CUDA.Wrappers;

internal sealed unsafe class CuTensorWorkspace : IDisposable
{
    internal readonly void* Pointer;
    internal readonly ulong Bytes;

    public CuTensorWorkspace(ulong bytes)
    {
        Pointer = CuArray.AllocateRaw((uint)bytes);
        Bytes = bytes;
    }

    public void Dispose()
    {
        CuArray.Free(Pointer);
    }
}