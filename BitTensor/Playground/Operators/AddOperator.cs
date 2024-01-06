using System.Runtime.Intrinsics;

namespace BitTensor.Playground.Operators;

using Vec = Vector256<float>;

public readonly struct AddOperator : IBinaryOperator<float>
{
    public static float Identity => 0;
    public static float Invoke(float a, float b) => a + b;
    public static Vec Invoke(Vec a, Vec b) => a + b;
}