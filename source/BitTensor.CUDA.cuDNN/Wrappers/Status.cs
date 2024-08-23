using System.Diagnostics;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

internal static class Status
{
    [StackTraceHidden]
    public static void EnsureIsSuccess(cudnnStatus_t status)
    {
        if (status != cudnnStatus_t.CUDNN_STATUS_SUCCESS)
            throw new CudnnException(status);
    }
}