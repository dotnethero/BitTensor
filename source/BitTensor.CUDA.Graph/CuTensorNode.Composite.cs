using System.Numerics;

namespace BitTensor.CUDA.Graph;

public static partial class CuTensorNode
{
    public static CuTensorNode<T> Softmax<T>(CuTensorNode<T> a) where T : unmanaged, IFloatingPoint<T>
    {
        var max = Max(a, [^1], keepDims: true);
        var ex = Exp(a - max);
        var sumex = Sum(ex, [^1], keepDims: true);
        return ElementwiseProduct(ex, Reciprocal(sumex));
    }
}