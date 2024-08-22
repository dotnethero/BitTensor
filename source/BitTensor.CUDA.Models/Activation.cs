using BitTensor.CUDA.Graph;

namespace BitTensor.CUDA.Models;

public static class Activation
{
    public static ActivationFunction<float> ReLU(float alpha) => t => Ops.ReLU(t, alpha);
    public static ActivationFunction<float> Softmax { get; } = Ops.Softmax;
}