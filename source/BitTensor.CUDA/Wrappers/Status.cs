using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

internal static class Status
{
    public static void EnsureIsSuccess(cutensorStatus_t status)
    {
        if (status != cutensorStatus_t.CUTENSOR_STATUS_SUCCESS)
            throw new CuTensorException(status);
    }

    public static void EnsureIsSuccess(cublasStatus_t status)
    {
        if (status != cublasStatus_t.CUBLAS_STATUS_SUCCESS)
            throw new CublasException(status);
    }

    public static void EnsureIsSuccess(curandStatus status)
    {
        if (status != curandStatus.CURAND_STATUS_SUCCESS)
            throw new CuRandException(status);
    }
}