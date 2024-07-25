namespace BitTensor.Abstractions;

public interface ITensor<T> where T : ITensor<T>
{
    static abstract T Zero { get; }
    static abstract T One { get; }

    static abstract T Add(T a, T b);
    static abstract T Add(T a, float b);
    static abstract T Add(float a, T b);
    static abstract T Mul(T a, T b);
    static abstract T Mul(T a, float b);
    static abstract T Mul(float a, T b);
    static abstract T Negate(T a);

    static virtual T operator +(T a, T b) => T.Add(a, b);
    static virtual T operator +(T a, float b) => T.Add(a, b);
    static virtual T operator +(float a, T b) => T.Add(a, b);
    static virtual T operator -(T a, T b) => T.Add(a, -b);
    static virtual T operator -(T a, float b) => T.Add(a, -b);
    static virtual T operator -(float a, T b) => T.Add(a, -b);
    static virtual T operator *(T a, T b) => T.Mul(a, b);
    static virtual T operator *(T a, float b) => T.Mul(a, b);
    static virtual T operator *(float a, T b) => T.Mul(a, b);
    static virtual T operator -(T a) => T.Negate(a);
}
