using System.Runtime.Intrinsics;

namespace BitTensor.Playground;

public interface IBinaryOperator<T>
{
    static abstract T Identity { get; }
    static abstract T Invoke(T a, T b);
    static abstract Vector256<T> Invoke(Vector256<T> a, Vector256<T> b);
}