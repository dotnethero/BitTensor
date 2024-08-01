using System.Numerics.Tensors;

namespace BitTensor.Operators;

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