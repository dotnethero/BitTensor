using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

internal static class Status
{
    public static void EnsureIsSuccess(curandStatus status)
    {
        if (status != curandStatus.CURAND_STATUS_SUCCESS)
            throw new CuRandException(status);
    }
}