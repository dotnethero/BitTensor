using ILGPU;

namespace BitTensor.CUDA.Kernels;

public static class CuKernelCode
{
    public static void Set(Index1D i, ArrayView<float> a, float value)
    {
        a[i] = value;
    }
    
    public static void Add(Index1D i, ArrayView<float> a, ArrayView<float> b, ArrayView<float> c)
    {
        c[i] = a[i] + b[i];
    }
    
    public static void Multiply(Index1D i, ArrayView<float> a, ArrayView<float> b, ArrayView<float> c)
    {
        c[i] = a[i] * b[i];
    }
}