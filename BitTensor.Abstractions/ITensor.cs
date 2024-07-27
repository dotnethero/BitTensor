namespace BitTensor.Abstractions;

public interface ITensor<T> where T : ITensor<T>
{
    internal static T Add(T a, T b) => a + b;

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

public interface IHasAllocator<out T> where T : ITensor<T>
{
    ITensorAllocator<T> Allocator { get; }
}
