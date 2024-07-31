using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA;

using static cudaRT;

internal static unsafe class CuMemory
{
    public static void CopyToHost(float* source, Span<float> destination, int count)
    {
        fixed (float* dp = destination)
            cudaMemcpy(dp, source, (uint)count * sizeof(float), cudaMemcpyKind.cudaMemcpyDeviceToHost);
    }

    public static void CopyToDevice(ReadOnlySpan<float> source, float* destination, int count)
    {
        fixed (float* sp = source)
            cudaMemcpy(destination, sp, (uint)count * sizeof(float), cudaMemcpyKind.cudaMemcpyHostToDevice);
    }
}