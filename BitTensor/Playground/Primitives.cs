using System.Runtime.CompilerServices;
using BitTensor.Playground.Operators;

namespace BitTensor.Playground;

public static class Primitives
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Add(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result) =>
        Operations.BinaryOperation<AddOperator>(a, b, result);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Multiply(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result) =>
        Operations.BinaryOperation<MultiplyOperator>(a, b, result);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Sum(ReadOnlySpan<float> a) =>
        Operations.AggregateUnary<IdentityOperator, AddOperator>(a);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b) =>
        Operations.AggregateBinary<MultiplyOperator, AddOperator>(a, b);
    
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DotTernary(ReadOnlySpan<float> a, ReadOnlySpan<float> b) =>
        Operations.AggregateTernary<MultiplyAddOperator>(a, b);
    
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DotFma(ReadOnlySpan<float> a, ReadOnlySpan<float> b) =>
        Operations.AggregateTernary<MultiplyAddFmaOperator>(a, b);
}
