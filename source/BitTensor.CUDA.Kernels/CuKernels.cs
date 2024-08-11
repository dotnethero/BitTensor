using ILGPU;
using ILGPU.Algorithms;

namespace BitTensor.CUDA.Kernels;

public static class CuKernels
{
    public static void Set(Index1D i, ArrayView<float> a, float value)
    {
        a[i] = value;
    }
    
    public static void Sigmoid(Index1D i, ArrayView<float> a)
    {
        a[i] = 1f / (1f + XMath.Exp(-a[i]));
    }
}