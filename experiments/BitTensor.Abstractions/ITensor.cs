namespace BitTensor.Abstractions;

public interface ITensor<T> where T : ITensor<T>
{
    static abstract T operator +(T a, T b);
    static abstract T operator +(T a, float b);
    static abstract T operator +(float a, T b);
    static abstract T operator -(T a, T b);
    static abstract T operator -(T a, float b);
    static abstract T operator -(float a, T b);
    static abstract T operator *(T a, T b);
    static abstract T operator *(T a, float b);
    static abstract T operator *(float a, T b);
    static abstract T operator -(T a);
}

public interface IMutableTensor<T> : ITensor<T> where T : ITensor<T>
{
    void ApplyOffset(T offset);
    void ApplyScale(T scale);
}
