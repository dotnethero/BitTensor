using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Internals;

using static cuBLAS;
using static cublasStatus_t;

internal readonly unsafe struct CublasScope : IDisposable
{
    internal readonly cublasContext* Context;

    public CublasScope()
    {
        cublasContext* handle;

        var status = cublasCreate_v2(&handle);
        if (status != CUBLAS_STATUS_SUCCESS)
            throw new InvalidOperationException(status.ToString());

        Context = handle;
    }

    public void Dispose()
    {
        cublasDestroy_v2(Context);
    }
}
