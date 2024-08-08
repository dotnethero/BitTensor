using System.Runtime.CompilerServices;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA;

using static cudaRT;

public static unsafe class CuArray
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static uint Bytes(int size) => (uint)size  * sizeof(float);
    
    public static void* AllocateBytes(uint bytes)
    {
        void* pointer;
        cudaMalloc(&pointer, bytes);
        return pointer;
    }

    public static float* Allocate(int size)
    {
        return (float*)AllocateBytes(Bytes(size));
    }

    public static float* Allocate(int size, float[] values)
    {
        var pointer = Allocate(size);
        CopyToDevice(values, pointer, size);
        return pointer;
    }
    
    public static void CopyToHost(float* source, Span<float> destination, int size)
    {
        fixed (float* dp = destination)
        {
            cudaMemcpy(dp, source, Bytes(size), cudaMemcpyKind.cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
        }
    }
    
    public static void CopyToDevice(ReadOnlySpan<float> source, float* destination, int size)
    {
        fixed (float* sp = source)
        {
            cudaMemcpy(destination, sp, Bytes(size), cudaMemcpyKind.cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
        }
    }

    public static void Free(void* pointer)
    {
        cudaFree(pointer);
    }
}
