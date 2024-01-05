using System.Numerics.Tensors;

namespace BitTensor.Core;

internal interface IBinaryOperator<T>
{
    static abstract T Execute(T a, T b);
    static abstract void Execute(ReadOnlySpan<T> a, T b, Span<T> result);
    static abstract void Execute(ReadOnlySpan<T> a, ReadOnlySpan<T> b, Span<T> result);
}

internal interface IAggregateOperator<T> : IBinaryOperator<T>
{
    static abstract T Identity { get; }
    static abstract T Aggregate(ReadOnlySpan<T> a);
}

internal readonly struct AddOperator : IAggregateOperator<float>
{
    public static float Identity => 0f;

    public static float Execute(float a, float b) => 
        a + b;

    public static void Execute(ReadOnlySpan<float> a, float b, Span<float> result) => 
        TensorPrimitives.Add(a, b, result);

    public static void Execute(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result) => 
        TensorPrimitives.Add(a, b, result);

    public static float Aggregate(ReadOnlySpan<float> a) => 
        TensorPrimitives.Sum(a);
}

internal readonly struct MultiplyOperator : IAggregateOperator<float>
{
    public static float Identity => 1f;

    public static float Execute(float a, float b) => 
        a * b;

    public static void Execute(ReadOnlySpan<float> a, float b, Span<float> result) => 
        TensorPrimitives.Multiply(a, b, result);

    public static void Execute(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result) => 
        TensorPrimitives.Multiply(a, b, result);

    public static float Aggregate(ReadOnlySpan<float> a) => 
        TensorPrimitives.Product(a);
}
