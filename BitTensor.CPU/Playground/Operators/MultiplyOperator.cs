using System.Runtime.Intrinsics;

namespace BitTensor.Playground.Operators;

using Vec = Vector256<float>;

public readonly struct MultiplyOperator : IBinaryOperator<float>
{
    public static float Identity => 1;
    public static float Invoke(float a, float b) => a * b;
    public static Vec Invoke(Vec a, Vec b) => a * b;
}