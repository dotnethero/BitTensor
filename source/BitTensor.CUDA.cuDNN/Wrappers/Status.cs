using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

internal static class Status
{
    public static void EnsureIsSuccess(cudnnStatus_t status)
    {
        if (status != cudnnStatus_t.CUDNN_STATUS_SUCCESS)
            throw new CudnnException(status);
    }
}