using System.Runtime.Intrinsics;

namespace BitTensor.Playground;

public interface IUnaryOperator<T>
{
    static abstract T Invoke(T a);
    static abstract Vector256<T> Invoke(Vector256<T> a);
}