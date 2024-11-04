using BitTensor.CUDA.Interop;
using BitTensor.CUDA.Wrappers;

namespace BitTensor.CUDA;

internal static class Status
{
    public static void EnsureIsSuccess(cudaError error)
    {
        if (error != cudaError.cudaSuccess)
            throw new CudaException(error);
    }
}