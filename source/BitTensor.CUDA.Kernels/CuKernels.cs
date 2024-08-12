using ILGPU;
using ILGPU.Algorithms;

namespace BitTensor.CUDA.Kernels;

public static class CuKernels
{
    public static void Set(Index1D i, ArrayView<float> a, float value)
    {
        a[i] = value;
    }
    
    public static void SigmoidForward(Index1D i, ArrayView<float> a, ArrayView<float> output)
    {
        output[i] = 1f / (1f + XMath.Exp(-a[i]));
    }
    
    public static void SigmoidBackward(Index1D i, ArrayView<float> grad, ArrayView<float> sigmoid, ArrayView<float> output)
    {
        output[i] = grad[i] * sigmoid[i] * (1f - sigmoid[i]);
    }
    
    public static void TanhForward(Index1D i, ArrayView<float> a, ArrayView<float> output)
    {
        output[i] = XMath.Tanh(a[i]);
    }

    public static void TanhBackward(Index1D i, ArrayView<float> grad, ArrayView<float> tanh, ArrayView<float> output)
    {
        output[i] = grad[i] * (1f - tanh[i] * tanh[i]);
    }
}
