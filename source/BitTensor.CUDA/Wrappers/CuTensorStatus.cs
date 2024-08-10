using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Wrappers;

internal static class CuTensorStatus
{
    public static void EnsureIsSuccess(cutensorStatus_t status)
    {
        if (status != cutensorStatus_t.CUTENSOR_STATUS_SUCCESS)
            throw new CuTensorException(status);
    }
}