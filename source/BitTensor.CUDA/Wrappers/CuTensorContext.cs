using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Operations;

namespace BitTensor.CUDA.Wrappers;

public unsafe class CuTensorContext : IDisposable
{
    internal readonly cutensorHandle* Handle;

    public CuTensorContext()
    {
        cutensorHandle* handle;

        var status = cuTENSOR.cutensorCreate(&handle);
        CuTensorStatus.EnsureIsSuccess(status);

        Handle = handle;
    }

    public CuTensorDescriptor CreateDescriptor(CuTensor a) => 
        new(this, a);
    
    public CuTensorDescriptor CreateDescriptor(CuTensor a, int[] modes) => 
        new(this, a, modes);
    
    public CuTensorContraction CreateContraction(
        CuTensorDescriptor a,
        CuTensorDescriptor b,
        CuTensorDescriptor c,
        CuTensorDescriptor d) => 
        new(this, a, b, c, d);
    
    public void Dispose()
    {
        cuTENSOR.cutensorDestroy(Handle);
    }
}