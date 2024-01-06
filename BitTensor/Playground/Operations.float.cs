using System.Runtime.CompilerServices;
using Elm = float;

namespace BitTensor.Playground;

public static partial class Operations
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void BinaryOperation<TOperator>(ReadOnlySpan<Elm> a, ReadOnlySpan<Elm> b, Span<Elm> result)
        where TOperator : IBinaryOperator<Elm> =>
        BinaryOperation<Elm, TOperator>(a, b, result);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Elm AggregateUnary<TOperator, TAggregate>(ReadOnlySpan<Elm> a)
        where TOperator : IUnaryOperator<Elm>
        where TAggregate : IBinaryOperator<Elm> =>
        AggregateUnary<Elm, TOperator, TAggregate>(a);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Elm AggregateBinary<TOperator, TAggregate>(ReadOnlySpan<Elm> a, ReadOnlySpan<Elm> b)
        where TOperator : IBinaryOperator<Elm>
        where TAggregate : IBinaryOperator<Elm> =>
        AggregateBinary<Elm, TOperator, TAggregate>(a, b);
    
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Elm AggregateTernary<TOperator>(ReadOnlySpan<Elm> a, ReadOnlySpan<Elm> b)
        where TOperator : ITernaryOperator<Elm> =>
        AggregateTernary<Elm, TOperator>(a, b);
}
