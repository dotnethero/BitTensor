namespace BitTensor.Abstractions;

public interface ITensor<T> where T : ITensor<T>
{
    static abstract T operator +(T a, T b);
    static abstract T operator *(T a, T b);
    static abstract T Sum(T a);
}