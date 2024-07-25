using static BitTensor.CUDA.Interop.cudaRT;

namespace BitTensor.CUDA;

internal static unsafe class CuAllocator
{
    public static float* Allocate(int size)
    {
        float* handle;
        cudaMalloc((void**)&handle, (uint)size * sizeof(float));
        return handle;
    }
    
    public static void Free(float* handle)
    {
        cudaFree(handle);
    }
}