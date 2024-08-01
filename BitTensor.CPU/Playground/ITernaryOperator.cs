using System.Runtime.Intrinsics;

namespace BitTensor.Playground;

public interface ITernaryOperator<T>
{
    static abstract T Identity { get; }
    static abstract T Invoke(T a, T b, T c);
    static abstract Vector256<T> Invoke(Vector256<T> a, Vector256<T> b, Vector256<T> c);
}
