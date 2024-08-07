namespace BitTensor.CUDA.Wrappers;

internal unsafe class CuTensorWorkspace : IDisposable
{
    internal readonly void* Pointer;
    internal readonly ulong Bytes;

    public CuTensorWorkspace(ulong bytes)
    {
        Pointer = CuArray.AllocateBytes((uint)bytes);
        Bytes = bytes;
    }

    public void Dispose()
    {
        CuArray.Free(Pointer);
    }
}