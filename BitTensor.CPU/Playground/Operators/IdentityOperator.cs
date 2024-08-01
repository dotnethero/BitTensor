using System.Runtime.Intrinsics;

namespace BitTensor.Playground.Operators;

public readonly struct IdentityOperator : IUnaryOperator<float>
{
    public static float Invoke(float a) => a;
    public static Vector256<float> Invoke(Vector256<float> a) => a;
}
