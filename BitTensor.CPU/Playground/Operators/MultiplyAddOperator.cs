using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace BitTensor.Playground.Operators;

using Vec = Vector256<float>;

public readonly struct MultiplyAddOperator : ITernaryOperator<float>
{
    public static float Identity => 0;
    public static float Invoke(float a, float b, float c) => a * b + c;
    public static Vec Invoke(Vec a, Vec b, Vec c) => a * b + c;
}

public readonly struct MultiplyAddFmaOperator : ITernaryOperator<float>
{
    public static float Identity => 0;
    public static float Invoke(float a, float b, float c) => a * b + c;
    public static Vec Invoke(Vec a, Vec b, Vec c) => Fma.MultiplyAdd(a, b, c);
}
