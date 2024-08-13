using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

internal static class Status
{
    public static void EnsureIsSuccess(cublasStatus_t status)
    {
        if (status != cublasStatus_t.CUBLAS_STATUS_SUCCESS)
            throw new CublasException(status);
    }
}